# Virtual-Glasses-Try-on
To Ensure the program is able to run on your local PC, ensure that the following libraries are up-to-date
1. opencv-pythonc2numpy (Updating opencv resolves both libraries)
2. Flask
3. werkzeug (I believe this is included with python as default and updated with each python release, so no need to worry)

Along with this, ensure all files are in the same Folders that you see in the repository.
Note there has been an issue where when the Repository is pulled the "static" folder is saved as "Static" locally.
Upon last checking this issue has been resolved but please be aware this may happen.

If you choose to upload you're own pair of glasses for testing, the app only allows for PNG images, and performs best with those with white backgrounds.

The application should default to launch at 127.0.0.1:5000, but double check where it launchs to in your VS code terminal, or whichever IDE you prefer, the application was created and tested in Visual Studio Code with Python 3.10.

For myself, I had to run the application via the F5 command in vs-code in debug mode, then selecting run as a flask application,  as the Flask run command wasn't working in my terminal.

Enjoy!
