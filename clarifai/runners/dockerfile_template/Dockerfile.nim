FROM nvcr.io/nim/meta/llama-3.1-8b-instruct:1.1.2 as build

FROM gcr.io/distroless/python3-debian12:debug


COPY --from=build /bin/bash /bin/rbash
COPY --from=build /bin/sh /bin/sh
COPY --from=build /bin/rsh /bin/rsh

# we have to overwrite the python3 binary that the distroless image uses
COPY --from=build /opt/nim/llm/.venv/bin/python3.10 /usr/bin/python3
COPY --from=build /opt/nim/llm/.venv/bin/python3.10 /usr/local/bin/python3.10

# also copy in all the lib files for it.
COPY --from=build /lib /lib
COPY --from=build /lib64 /lib64
COPY --from=build /usr/lib/ /usr/lib/
COPY --from=build /usr/local/lib/ /usr/local/lib/
# ldconfig is needed to update the shared library cache so system libraries (like CUDA) can be found
COPY --from=build /usr/sbin/ldconfig /sbin/ldconfig
COPY --from=build /usr/sbin/ldconfig.real /sbin/ldconfig.real
COPY --from=build /etc/ld.so.conf /etc/ld.so.conf
COPY --from=build /etc/ld.so.cache /etc/ld.so.cache
COPY --from=build /etc/ld.so.conf.d/ /etc/ld.so.conf.d/

# COPY NIM files
COPY  --from=build /opt /opt
COPY  --from=build /etc/nim /etc/nim

# Set environment variables to use the nim libraries and python
ENV PYTHONPATH=${PYTHONPATH}:/opt/nim/llm/.venv/lib/python3.10/site-packages:/opt/nim/llm
ENV PATH="/opt/nim/llm/.venv/bin:/opt/hpcx/ucc/bin:/opt/hpcx/ucx/bin:/opt/hpcx/ompi/bin:$PATH"

ENV LD_LIBRARY_PATH="/opt/hpcx/ucc/lib/ucc:/opt/hpcx/ucc/lib:/opt/hpcx/ucx/lib/ucx:/opt/hpcx/ucx/lib:/opt/hpcx/ompi/lib:/opt/hpcx/ompi/lib/openmpi:/opt/nim/llm/.venv/lib/python3.10/site-packages/tensorrt_llm/libs:/opt/nim/llm/.venv/lib/python3.10/site-packages/nvidia/cublas/lib:/opt/nim/llm/.venv/lib/python3.10/site-packages/tensorrt_libs:/opt/nim/llm/.venv/lib/python3.10/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH"

ENV LIBRARY_PATH=/opt/hpcx/ucc/lib:/opt/hpcx/ucx/lib:/opt/hpcx/ompi/lib:$LIBRARY_PATH

ENV CPATH=/opt/hpcx/ompi/include:/opt/hpcx/ucc/include:/opt/hpcx/ucx/include:$CPATH
ENV LLM_PROJECT_DIR=/opt/nim/llm

# Set environment variables for MPI
ENV OMPI_HOME=/opt/hpcx/ompi
ENV HPCX_MPI_DIR=/opt/hpcx/ompi
ENV MPIf_HOME=/opt/hpcx/ompi
ENV OPAL_PREFIX=/opt/hpcx/ompi

# Set environment variables for UCC
ENV UCC_DIR=/opt/hpcx/ucc/lib/cmake/ucc
ENV UCC_HOME=/opt/hpcx/ucc
ENV HPCX_UCC_DIR=/opt/hpcx/ucc
ENV USE_UCC=1
ENV USE_SYSTEM_UCC=1

# Set environment variables for HPC-X
ENV HPCX_DIR=/opt/hpcx
ENV HPCX_UCX_DIR=/opt/hpcx/ucx
ENV HPCX_MPI_DIR=/opt/hpcx/ompi

# Set environment variables for UCX
ENV UCX_DIR=/opt/hpcx/ucx/lib/cmake/ucx
ENV UCX_HOME=/opt/hpcx/ucx

ENV HOME=/opt/nim/llm

# ln is needed to create symbolic links (needed by nvidia-container-runtime)
COPY --from=build /usr/bin/ln /usr/bin/ln

# Run ldconfig in the build stage to update the library cache else CUDA libraries won't be found
RUN ldconfig -v

SHELL ["/bin/rbash", "-c"]
