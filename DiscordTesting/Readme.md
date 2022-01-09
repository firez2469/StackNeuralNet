# Discord Testing
Program by: Marco S. Hampel
## Description
This program will run neural network which is trained to identify if a message is from a given person by analyzing their discord conversation. The program does not load the discord conversation
itself, but reads it off of a text file which is loaded into this text file by a seperate program.

## Techncal Description
This file loads a text file with the a list of discord messages. Each message is converted to utf-8 integer values which are then converted into 1d vector used for the input layer of the 
stack neural network. The loaded text file also contains the discord username of the user who used it. The program will check if the person we want our neural net to recognize
is the author of each message and assign a 1 if they are the author, and a 0 if they are not. This 1D array of 1's and 0's becomes the expected output layer for each message.

Since *Stack Neural Net* only supports a binary output layer the program the usecase is limited to identifying if the given message is from a specific person or not the specific
person. After the Neural net loads all the training data using the *.fit()* function, the neural net is trained 500 times with a 0.3 learning rate. 

Once the neural network is trained the data is tested with sample strings that can either correspond or not correspond to the targeted user.

The Text File the conversation is loaded off of is of the given format:
    
    Person1|Hello this is my text conversaation
    Person2|TextConversation
    Person3|What's up my guys? How's it going?
    ...
    
 The author is the first field, split by the ```|``` followed by the message. ```\n``` or ```\r``` are used to parse each message
