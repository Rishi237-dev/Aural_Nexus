# Deploying the Streamlit UI

## Option 1: Streamlit Community Cloud
1. Push this repository to GitHub.
2. Open https://share.streamlit.io/.
3. Click New app.
4. Select the repository and branch.
5. Set the main file to app.py.
6. Deploy.

## Option 2: Render
1. Create a new Web Service on Render.
2. Connect this GitHub repository.
3. Use the following settings:
   - Build Command: pip install -r requirements.txt
   - Start Command: streamlit run app.py --server.port $PORT --server.address 0.0.0.0

## Option 3: Railway / Fly.io
Use the same app.py entry point and install requirements from requirements.txt.
