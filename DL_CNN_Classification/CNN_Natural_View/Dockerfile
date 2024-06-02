FROM ubuntu

WORKDIR /app

RUN apt-get update -y
RUN apt-get install -y python3 python3-pip python3-dev build-essential hdf5-tools libgl1 libgtk2.0-dev 
RUN pip install -y streamlit
# RUN pip3 install streamlit

COPY . /app

EXPOSE 8501
CMD ['python3','streamlit','run','APP_CNN_Natural_View']
# RUN pip install pillow matplotlib numpy tensorflow==2.13.0 -y