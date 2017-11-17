#ifndef user_heat_world_hpp
#define user_heat_world_hpp

#include "puzzler/puzzles/heat_world.hpp"

#include <stdexcept>
#include <cmath>
#include <cstdint>
#include <memory>
#include <cstdio>
#include <string>
#include <cstdlib>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include <fstream>
#include <streambuf>

class HeatWorldProvider
  : public puzzler::HeatWorldPuzzle
{
private:
	std::string LoadSource(const char *fileName) const {

		// TODO : Don't forget to change your_login here
		std::string baseDir="provider";
		if(getenv("HPCE_CL_SRC_DIR")){
			baseDir=getenv("HPCE_CL_SRC_DIR");
		}
	
		std::string fullName=baseDir+"/"+fileName;
	
		// Open a read-only binary stream over the file
		std::ifstream src(fullName, std::ios::in | std::ios::binary);
		if(!src.is_open())
			throw std::runtime_error("LoadSource : Couldn't load cl file from '"+fullName+"'.");
	
		// Read all characters of the file into a string
		return std::string(
			(std::istreambuf_iterator<char>(src)), // Node the extra brackets.
		std::istreambuf_iterator<char>()
		);
	}

public:
  HeatWorldProvider()
  {}
	
	virtual void Execute(
			   puzzler::ILog *log,
			   const puzzler::HeatWorldInput *input,
					puzzler::HeatWorldOutput *output
			   ) const override
	{
		//choosing a platform

		std::vector<cl::Platform> platforms;
	
		cl::Platform::get(&platforms);
		if(platforms.size()==0)
		throw std::runtime_error("No OpenCL platforms found.");	

		std::cerr<<"Found "<<platforms.size()<<" platforms\n";
		for(unsigned i=0;i<platforms.size();i++){
			std::string vendor=platforms[i].getInfo<CL_PLATFORM_VENDOR>();
			std::cerr<<"  Platform "<<i<<" : "<<vendor<<"\n";
		}
	
		int selectedPlatform=0;
		if(getenv("HPCE_SELECT_PLATFORM")){
			selectedPlatform=atoi(getenv("HPCE_SELECT_PLATFORM"));
		}
		std::cerr<<"Choosing platform "<<selectedPlatform<<"\n";
		cl::Platform platform=platforms.at(selectedPlatform);
	   
		//end of platform choosing

		//choosing a device
	
		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);	
		if(devices.size()==0){
			throw std::runtime_error("No opencl devices found.\n");
		}
	
		std::cerr<<"Found "<<devices.size()<<" devices\n";
		for(unsigned i=0;i<devices.size();i++){
			std::string name=devices[i].getInfo<CL_DEVICE_NAME>();
			std::cerr<<"  Device "<<i<<" : "<<name<<"\n";
		}

		int selectedDevice=0;
		if(getenv("HPCE_SELECT_DEVICE")){
			selectedDevice=atoi(getenv("HPCE_SELECT_DEVICE"));
		}
		std::cerr<<"Choosing device "<<selectedDevice<<"\n";
		cl::Device device=devices.at(selectedDevice);
	
		//end of device choosing

		cl::Context context(devices);
		std::string kernelSource=LoadSource("heat_world.cl");

		cl::Program::Sources sources;
		sources.push_back(std::make_pair(kernelSource.c_str(), kernelSource.size()+1));

		cl::Program program(context, sources);
		try{
			program.build(devices);		
		}catch(...){
		  for(unsigned i=0;i<devices.size();i++){
		    std::cerr<<"Log for device "<<devices[i].getInfo<CL_DEVICE_NAME>()<<":\n\n";
		    std::cerr<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[i])<<"\n\n";
		  }
		  throw;
		}		
		
      		  unsigned n=input->n;
      
      		  float outer=input->alpha;
	          float inner=1-outer/4;
      
      		  const auto &properties=input->properties;
      		  auto state=input->state;
      		  std::vector<float> buffer(n*n);


		//Allocating buffers
		size_t cbBuffer=4*n*n;
		cl::Buffer buffProperties(context, CL_MEM_READ_ONLY, cbBuffer);
		cl::Buffer buffState(context, CL_MEM_READ_WRITE, cbBuffer);
		cl::Buffer buffBuffer(context, CL_MEM_READ_WRITE, cbBuffer);

		cl::Kernel kernel(program, "kernel_xy");
		//end of allocating buffers

		  std::vector<uint32_t> packed(n*n,0);	
		  for(unsigned y=0;y<n;y++){
		    for(unsigned x=0;x<n;x++){
		      unsigned P_index=n*y+x;
		      packed[P_index]=properties[P_index];
		      if(!( (packed[P_index] & Cell_Insulator) || (packed[P_index] & Cell_Fixed) ) ){
		        if(properties[P_index+1] & Cell_Insulator){
			  packed[P_index]=packed[P_index]+4; //bit 2 for right
			}
			if(properties[P_index-1] & Cell_Insulator){
			  packed[P_index]=packed[P_index]+8; //bit 3 for left
			}
			if(properties[P_index+n] & Cell_Insulator){
			  packed[P_index]=packed[P_index]+16; //bit 4 for below
			}
			if(properties[P_index-n] & Cell_Insulator){
			  packed[P_index]=packed[P_index]+32; //bit 5 for above
			}
		      }
		    }
		  }

      		  log->Log(puzzler::Log_Verbose, [&](std::ostream &dst){
        	    for(unsigned y=0; y<n; y++){
          	      for(unsigned x=0; x<n; x++){
                        dst<<properties[y*n+x];
          	      }
          	      dst<<"\n";
        	    }
      		  });

		//Setting kernel parameters
		kernel.setArg(0, inner);
		kernel.setArg(1, outer);
		kernel.setArg(2, buffProperties);
		kernel.setArg(3, buffState);
		kernel.setArg(4, buffBuffer);
		//end of setting kernel parameters
	
		//Creating a command queue
		cl::CommandQueue queue(context, device);
		//end of creating a command queue
	
		//Copying over fixed data packed
		queue.enqueueWriteBuffer(buffProperties, CL_TRUE, 0, cbBuffer, &packed[0]);
		//end of copying over fiexed data
	

		queue.enqueueWriteBuffer(buffState, CL_TRUE, 0, cbBuffer, &state[0]);
      
      		for(unsigned t=0;t<n;t++){
        	  log->LogDebug("Time step %d", t);

       		  //
		  cl::NDRange offset(0,0);

		  cl::NDRange globalSize(n,n);

		  cl::NDRange localSize=cl::NullRange;

		  kernel.setArg(3,buffState);
		  kernel.setArg(4,buffBuffer);

		  queue.enqueueNDRangeKernel(kernel, offset, globalSize, localSize);

		  queue.enqueueBarrier();
		  //
		
		  std::swap(buffState, buffBuffer);        
	        }
	  	
		queue.enqueueReadBuffer(buffState, CL_TRUE, 0, cbBuffer, &state[0]);

	      log->Log(puzzler::Log_Verbose, [&](std::ostream &dst){
		dst<<std::fixed<<std::setprecision(2);
		for(unsigned y=0; y<n; y++){
		  for(unsigned x=0; x<n; x++){
		    dst<<" "<<std::setw(6)<<state[y*n+x];
		  }
		  dst<<"\n";
		}
	      });

      	      output->state=state;
    	    
	}

};

#endif
