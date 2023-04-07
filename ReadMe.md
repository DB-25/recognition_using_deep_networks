# Project 5

Authors: Yalala Mohit & Dhruv Kamalesh Kumar
Late days used:
        3 late days have been used.

OS : Windows, VS code.

How to run:
        execute each file as "python filename.py" to execute each task.
        
Files:
    main.py contains all the tasks for Task 1.
    task1G.py contains all the code for Task 1G
    task1F.py contains all the code for Task 1F

    task2.py contains task 2.
    task3.py contains task 3.
    task4.py contains task 4.

    extension_task2.py is the extension 1. 
            There are many pre-trained networks available in the PyTorch package.                
            Try loading one and evaluate its first couple of convolutional layers as in task 2.

    extensionFrozenFilter.py is the extension 2.
            Replace the first layer of the MNIST network with a filter bank of your choosing (e.g. Gabor filters) and retrain the rest of the network, holding the first layer constant. How does it do?

    videoDigitRecognition.py is the extension 3.
            Build a live video digit recognition application using the trained network.

    videoMultiDigitRecognition.py is the extension 4.
            Build a live video Multiple digit recognition application using the trained network.


Extensions:
    5 extensions have been done.
        1. Evaluated First two conv filters on VGG16 dataset.
        2. Replaced First conv layer weights with 3 filters
            Gabor filter
            Laplacian Filter
            Gaussian filter
            and having them freezed, trained and tested the network. Gaussian filter works the best.
        3.  Single digit video recognition Application
        4.  Multiple digit video recognition application
        5.  Trained 1024 combination of models, and saved their confusion matrices and hyperparameters 
            in a dataframe, and saved them as csv, and json.
