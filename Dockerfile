FROM python:3.12-slim

# pyshtools needs a Fortran runtime and FFTW; everything else is wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gfortran libfftw3-dev libopenblas-dev git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY experiments/ experiments/
COPY figures/ figures/

# One BLAS thread by default. Every experiment reports a cost per model day,
# and a number measured while the library silently used every core is not
# comparable with anything. Override at run time if you want the speed.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONPATH=/work/experiments \
    PYTHONUNBUFFERED=1 \
    RESULTS_DIR=/work/results

CMD ["python", "-c", "print('pick a service: docker compose run --rm exp01 ... see README')"]
