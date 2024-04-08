
# FROM tells Docker which base image to use for this build 
FROM python:3.10
# COPY is used to copy tftest.py from the local machine 
# to a location inside the image

# Update package lists and install libgl1-mesa-glx
COPY ca_main.py /tmp
COPY ca_prompts.py /tmp
COPY sara.csv /tmp

# COPY data /tmp/data


# ADD data /tmp/data
# ENV can be used to set environment variables
ENV TRANSFORMERS_CACHE /nfs
# RUN is used to execute a command in the image

# WORKDIR configures the current working directory that 
# the CMD will be executed within. Since the COPY
# command above puts the script under /tmp, we'll set 
# it to the same location so we can run the script with 
# no path prefix
WORKDIR /tmp

RUN python3 -m pip install --upgrade pip
RUN pip3 install wheel
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install transformers
RUN pip install pillow
RUN pip install pandas
# RUN pip install ir_datasets
RUN pip install gensim
RUN pip install scipy==1.12
RUN pip install accelerate
RUN pip install bitsandbytes


# CMD defines the command that containers will run when # created from this image
# CMD ["python", "ca_main.py"]
# CMD ["python", "ca_main.py", "test-mist", "cpu", "0", "1"]
# pipeline(mname='test-mist', d='cpu', prompt_no=0, n=1)

