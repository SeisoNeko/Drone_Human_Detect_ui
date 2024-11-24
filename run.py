import os

if __name__ == '__main__':
    try:
        r = os.system('streamlit run main.py --server.maxUploadSize 10000')
        if r != 0:
            os.system('python -m streamlit run main.py --server.maxUploadSize 10000')
    except:
        pass