import subprocess

def run_streamlit():
    # Execute the Streamlit app via subprocess
    subprocess.run(["streamlit", "run", "streamlit_app.py"])

def main():
    run_streamlit

if __name__ == "__main__":
    main()