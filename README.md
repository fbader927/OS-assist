# LLM Vision OS

LLM Vision OS is a Python-based application that utilizes Gemini 1.5 Flash API for image analysis and and Google Speech for speech synthesis. When you click 'start' it will take screenshots every few seconds for Gemini Flash to analyze. You can ask questions and get answers related to anything you're viewing on the screen. 


![image](https://github.com/fbader927/OS-assist/assets/50185837/28645445-172b-4dfc-a8cf-8ecd61f315e2)



## Features

- Capture and analyze screenshots at specified intervals
- Real-time speech recognition and synthesis
- Export logs of analysis
- Simple GUI

### Prerequisites

- Python 3.10 or later
- Required Python packages (see `requirements.txt`)
- Google Cloud account with `Generative Language API` and `Cloud Text-to-Speech API` both enabled
- Google Cloud API set as environment variable with name `GOOGLE_API_KEY`

## Note
- Screenshots will be taken and processed every 2 seconds by default. This can be changed in the interface. 
