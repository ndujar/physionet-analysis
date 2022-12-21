FROM jupyter/datascience-notebook
RUN wget https://physionet.org/static/published-projects/ptbdb/ptb-diagnostic-ecg-database-1.0.0.zip
RUN git clone https://github.com/MIT-LCP/wfdb-python.git
RUN unzip ptb-diagnostic-ecg-database-1.0.0.zip -d wfdb-python/
RUN rm ptb-diagnostic-ecg-database-1.0.0.zip
RUN pip install plotly
RUN pip install wfdb
RUN pip install tslearn
RUN pip install heartpy

