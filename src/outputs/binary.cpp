//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file binary.cpp
//! \brief writes output data in binary format, which simply consists of each MeshBlock
//! written contiguously in order of "gid" in binary format.

#include <sys/stat.h>  // mkdir

#include <cstdio>      // fwrite(), fclose(), fopen(), fnprintf(), snprintf()
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm> // min

#include "athena.hpp"
#include "globals.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// Constructor: also calls BaseTypeOutput base class constructor

MeshBinaryOutput::MeshBinaryOutput(ParameterInput *pin, Mesh *pm, OutputParameters op) :
  BaseTypeOutput(pin, pm, op) {
  // create directories for outputs
  // useful for mpiio-based outputs because on some supercomputers you may need to
  // set different stripe counts depending on whether mpiio is used in order to
  // achieve the best performance and not to crash the filesystem
  mkdir("bin",0775);
  bool single_file_per_rank = op.single_file_per_rank;
  if (single_file_per_rank) {
    char rank_dir[20];
    std::snprintf(rank_dir, sizeof(rank_dir), "bin/rank_%08d/", global_variable::my_rank);
    mkdir(rank_dir, 0775);
  }
}

//----------------------------------------------------------------------------------------
//! \fn std::string MeshBinaryOutput::FileNameForNumber(int number)
//  \brief "bin/file_basename" + "." + "file_id" + "." + XXXXX + ".bin", where XXXXX is
//   the 5-digit file number.  Used both to write the file and to report, at startup,
//   which existing files a resumed run is about to supersede.

std::string MeshBinaryOutput::FileNameForNumber(int number) const {
  char digits[7];
  std::snprintf(digits, sizeof(digits), ".%05d", number);
  std::string dir("bin/");
  if (out_params.single_file_per_rank) {
    char rank_dir[20];
    std::snprintf(rank_dir, sizeof(rank_dir), "rank_%08d/", global_variable::my_rank);
    dir += rank_dir;
  }
  return dir + out_params.file_basename + "." + out_params.file_id + digits + ".bin";
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBinaryOutput:::WriteOutputFile(Mesh *pm)
//  \brief Cycles over all MeshBlocks and writes OutputData in binary format
//   All MeshBlocks are written to the same file.

void MeshBinaryOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  problem_runtime::StoreRuntimeMetadata(pin);
  if (pm != nullptr && pm->pgen != nullptr && pm->pgen->user_metadata_func != nullptr) {
    pm->pgen->user_metadata_func(pin, pm);
  }

  // check if slicing
  bool bin_slice = (out_params.slice1 || out_params.slice2 || out_params.slice3);

  bool single_file_per_rank = out_params.single_file_per_rank;
  const bool use_double = out_params.binary_double;
  const std::size_t variable_size = use_double ? sizeof(double) : sizeof(float);
  const std::string fname = FileNameForNumber(out_params.file_number);

  IOWrapper binfile;
  std::size_t header_offset=0;
  binfile.Open(fname.c_str(), IOWrapper::FileMode::write, single_file_per_rank);

  // Basic parts of the format:
  // 1. Size of the header
  // 2. Current time
  // 3. List of variables in the file
  // 4. Header (input file information)
  {
    std::stringstream msg;
    const int time_precision = std::numeric_limits<Real>::max_digits10 - 1;
    msg << "Athena binary output version=1.1" << std::endl
        // preheader size includes "size of preheader" line up to "number of variables"
        << "  size of preheader=5" << std::endl
        << std::scientific << std::setprecision(time_precision)
        << "  time=" << pm->time << std::endl
        << "  cycle=" << pm->ncycle << std::endl
        << "  size of location=" << sizeof(Real) << std::endl
        << "  size of variable=" << variable_size << std::endl
        << "  number of variables=" << outvars.size() << std::endl
        << "  variables:  ";
    for (size_t n=0; n<outvars.size(); n++) {
      msg << outvars[n].label.c_str() << "  ";
    }
    msg << std::endl;
    if (global_variable::my_rank == 0 || single_file_per_rank) {
      binfile.Write_any_type(msg.str().c_str(),msg.str().size(),"byte",
                             single_file_per_rank);
    }
    header_offset += msg.str().size();
  }
  {
    std::stringstream msg;
    // prepare the input parameters
    std::stringstream ost;
    pin->ParameterDump(ost);
    std::string sbuf=ost.str();
    msg << "  header offset=" << sbuf.size()*sizeof(char)  << std::endl;
    if (global_variable::my_rank == 0 || single_file_per_rank) {
      binfile.Write_any_type(msg.str().c_str(),msg.str().size(),"byte",
                             single_file_per_rank);
      binfile.Write_any_type(sbuf.c_str(),sbuf.size(),"byte", single_file_per_rank);
    }
    header_offset += sbuf.size()*sizeof(char);
    header_offset += msg.str().size();
  }
#if MPI_PARALLEL_ENABLED
  if (!single_file_per_rank) {
    // In shared-file mode only rank 0 writes the header, so every rank must use
    // rank 0's header length when computing its data offsets.
    std::uint64_t header_offset_u64 = static_cast<std::uint64_t>(header_offset);
    MPI_Bcast(&header_offset_u64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    header_offset = static_cast<std::size_t>(header_offset_u64);
  }
#endif

  //  5. Data.  An arbitrary number of scalars and vectors can be written (every element
  //  of the outvars vector), all in binary floats format

  int nout_vars = outvars.size();
  int nout_mbs = outmbs.size();
  int cells = 0;
  if (nout_mbs > 0) {
    int nout1 = outmbs[0].oie - outmbs[0].ois + 1;
    int nout2 = outmbs[0].oje - outmbs[0].ojs + 1;
    int nout3 = outmbs[0].oke - outmbs[0].oks + 1;
    cells = nout1*nout2*nout3;
  }

  // ois, oie, ojs, oje, oks, oke + il1, il2, il3, level +
  // x1min, x1max, x2min, x2max, x3min, x3max + data
  std::size_t data_size = 10*sizeof(int32_t) + 6*sizeof(Real)
                        + (cells*nout_vars)*variable_size;

  int ns_mbs = 0;
  for (int r = 0; r < global_variable::my_rank; ++r) {
    ns_mbs += noutmbs[r];
  }
  int nb_mbs = nout_mbs;

  // allocate 1D vector of floats used to convert and output data
  char *data = new char[nb_mbs*data_size];
  float *single_data = new float[cells];
  double *double_data = use_double ? new double[cells] : nullptr;

  // Loop over MeshBlocks
  for (int m=0; m<nout_mbs; ++m) {
    char *pdata=&(data[m*data_size]);
    LogicalLocation loc = pm->lloc_eachmb[outmbs[m].mb_gid];
    int &ois = outmbs[m].ois;
    int &oie = outmbs[m].oie;
    int &ojs = outmbs[m].ojs;
    int &oje = outmbs[m].oje;
    int &oks = outmbs[m].oks;
    int &oke = outmbs[m].oke;

    // output indexing for MB
    int32_t nx = (int32_t)(ois);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);
    nx = (int32_t)(oie);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);
    nx = (int32_t)(ojs);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);
    nx = (int32_t)(oje);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);
    nx = (int32_t)(oks);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);
    nx = (int32_t)(oke);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);

    // logical location lx1, lx2, lx3
    nx = (int32_t)(loc.lx1);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);
    nx = (int32_t)(loc.lx2);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);
    nx = (int32_t)(loc.lx3);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);

    // physical refinement level
    nx = (int32_t)(loc.level-pm->root_level);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);

    // coordinate location
    Real xv = outmbs[m].x1min;
    memcpy(pdata,&(xv),sizeof(xv));
    pdata+=sizeof(xv);
    xv = outmbs[m].x1max;
    memcpy(pdata,&(xv),sizeof(xv));
    pdata+=sizeof(xv);
    xv = outmbs[m].x2min;
    memcpy(pdata,&(xv),sizeof(xv));
    pdata+=sizeof(xv);
    xv = outmbs[m].x2max;
    memcpy(pdata,&(xv),sizeof(xv));
    pdata+=sizeof(xv);
    xv = outmbs[m].x3min;
    memcpy(pdata,&(xv),sizeof(xv));
    pdata+=sizeof(xv);
    xv = outmbs[m].x3max;
    memcpy(pdata,&(xv),sizeof(xv));
    pdata+=sizeof(xv);

    // output variables
    float tmp_data;
    for (int n=0; n<nout_vars; n++) {
      int cnt=0;
      for (int k=oks; k<=oke; k++) {
        for (int j=ojs; j<=oje; j++) {
          for (int i=ois; i<=oie; i++) {
            if (use_double) {
              double_data[cnt] = static_cast<double>(outarray(n,m,k-oks,j-ojs,i-ois));
            } else {
              tmp_data = static_cast<float>(outarray(n,m,k-oks,j-ojs,i-ois));
              single_data[cnt] = tmp_data;
            }
            cnt++;
          }
        }
      }
      memcpy(pdata,use_double ? static_cast<void *>(double_data) :
                               static_cast<void *>(single_data),cells*variable_size);
      pdata+=cells*variable_size;
    }
  }

  // now write binary data
  if (bin_slice) {
    std::vector<int> rank_offset(global_variable::nranks, 0);
    std::partial_sum(noutmbs.begin(),std::prev(noutmbs.end()),
                     std::next(rank_offset.begin()));
    std::size_t myoffset = header_offset+data_size*rank_offset[global_variable::my_rank];

    if (single_file_per_rank) {
      myoffset = header_offset;  // Reset offset for individual files
    }

    if (noutmbs_min > 0) {
      binfile.Write_any_type_at_all(data,(data_size*nout_mbs),myoffset,"byte",
                                    single_file_per_rank);
    } else {
      if (nout_mbs > 0) {
        binfile.Write_any_type_at(data,(data_size*nout_mbs),myoffset,"byte",
                                    single_file_per_rank);
      }
    }
  } else {
    // check if elements larger than 2^31
  if (data_size*nb_mbs<=2147483648) {
      // now write binary data in parallel
      std::size_t myoffset = header_offset;
      if (!single_file_per_rank) {
        myoffset += data_size*ns_mbs;
      }
      binfile.Write_any_type_at_all(data,(data_size*nb_mbs),myoffset,"byte",
                                    single_file_per_rank);
    } else {
      // write data over each MeshBlock sequentially and in parallel
      // calculate max/min number of MeshBlocks across all ranks
      for (int m=0;  m<noutmbs_max; ++m) {
        char *pdata=&(data[m*data_size]);
        std::size_t myoffset = header_offset + data_size*m;
        if (!single_file_per_rank) {
          myoffset += data_size*ns_mbs;
        }
        // every rank has a MB to write, so write collectively
        if (m < noutmbs_min) {
          if (binfile.Write_any_type_at_all(pdata,(data_size),myoffset,"byte",
                                              single_file_per_rank) != data_size) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "binary data not written correctly to binary file, "
                << "binary file is broken." << std::endl;
            exit(EXIT_FAILURE);
          }
        // some ranks are finished writing, so use non-collective write
        } else if (m < nout_mbs) {
          if (binfile.Write_any_type_at(pdata,(data_size),myoffset,"byte",
                                          single_file_per_rank) != data_size) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                 << std::endl << "binary data not written correctly to binary file, "
                 << "binary file is broken." << std::endl;
            exit(EXIT_FAILURE);
          }
        }
      }
    }
  }

  // close the output file and clean up ptrs to data
  binfile.Close(single_file_per_rank);
  delete [] data;
  delete [] single_data;
  delete [] double_data;

  // increment counters
  out_params.file_number++;
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  AdvanceOutputTime(pm, pin);

  return;
}
