# REAL-TIME-GESTURE-TO-SPEECH-A-CNN-APPROACH
This project presents a Real-Time Sign Language Translator using Deep Learning (CNN) and OpenCV to detect and translate 250 hand gesture signs into text and speech. The system bridges communication gaps for the deaf and dumb community with accurate sign recognition and Text-to-Speech (TTS) output.

# EXECUTION STEPS
 
• STEP 1: Save all your files (app.py, main.py, model files, src/folder, etc.) are in a single project folder. 

• STEP 2: In step 2, launch the VScode IDE and open the project folder in the IDE using File → Open Folder. 

• STEP 3: Install all required Python libraries, which are mentioned in the file requirements.txt in the project folder, by using the given command in the VS Code terminal

    pip install -r requirements.txt 
                                                                      
• STEP 4: Run the Streamlit app (app.py), which provides a web-based interface with speech support. Use the below command in Terminal to run it. The interface will open at the default web browser in 
                                                                            (http://localhost:8501).  
                                                                            
    python -m streamlit run app.py 
 
• STEP 5: If u want to test hand sign detection using the camera without the Streamlit interface, use the command below.

    python main.py 
                                               
• STEP 6: To interrupt or stop the process, use CTRL+C or type Q in the terminal. 
