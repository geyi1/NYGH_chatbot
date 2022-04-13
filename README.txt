Codebase for NYGH Chatbot

The code is split into two different secition: Training and Deployment

Further explanations are enclosed in each python files

requirements.txt: contains dependencies for the codebase, some might not be nesssary later on, depends on implmentation

Training Section:

Training codes are included in the training folder:
- data_augmentation.py: Used for data augmentation, incorparate nlpaug from Github to generate more data. TOUSE: Provide a NYGH.json that contains all the cetegories and patterns to be augmented and run python3 data_augmentation.py
- preprocessing.py: Helper used to clean raw data
- train_new: Use augmented_data.json generated from data_augmentation.py to train a chatbot model and save the model in the model folder


Deployment Section:
- deploy.py: Run deploy.py to host chatbot UI on localhost and testing. Folders templates and static are being used in deploy.py to support user interface. TOUSE: python3 deploy.py
- predict.py: Helper used by deploy.py to utilize trained model and predict answer

NYGH.json and augmented_data.json will be required to run above code. A example of NYGH.json is provided in the directory.



FOR REMOTE DEPLOYMENT:

We have used heroku on testing remote deployment of the chatbot, the codebase here is too large for heroku to compile and deploy.

I have created a separate github page here: https://github.com/geyi1/NYGH_chatbot_only for remote deployment, this repository is bascially the same, but without all the training code

To deploy with latest trained model, copy the newest chatbot_model_{}.h5, augmented_data.json, words.pkl, classes.pkl into the repository, and you can try on your own heroku account to deploy remotely

