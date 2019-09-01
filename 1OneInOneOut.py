from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding



# source text
#data = """ Jack and Jill went up the hill\n
		#To fetch a pail of water\n
		#Jack fell down and broke his crown\n
		#And Jill came tumbling after\n """
data = open('data.txt').read()


#data preparation
#############################################################
##############################################################
##############################################################


# integer encode text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]

#print(encoded)
#[2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 2, 14, 15, 1, 16, 17, 18, 1, 3, 19, 20, 21]

#We will need to know the size of the vocabulary later for both defining the word embedding layer
# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)




#Next, we need to create sequences of words to fit the model with one word as input and one word as output.
# create word -> word sequences
sequences = list()
for i in range(1, len(encoded)):
	sequence = encoded[i-1:i+1]
	sequences.append(sequence)

#print('Total Sequences: %d' % len(sequences))
#24

#print(sequences)
#[[2, 1], [1, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12],
# [12, 13], [13, 2], [2, 14], [14, 15], [15, 1], [1, 16], [16, 17], [17, 18], [18, 1],
# [1, 3], [3, 19], [19, 20], [20, 21]]




# split into X and y elements
sequences = array(sequences)
X, y = sequences[:,0],sequences[:,1]

#print(X)
#print(y)
#X=[ 2  1  3  4  5  6  7  8  9 10 11 12 13  2 14 15  1 16 17 18  1  3 19 20]
#y=[ 1  3  4  5  6  7  8  9 10 11 12 13  2 14 15  1 16 17 18  1  3 19 20 21]




#We will fit our model to predict a probability distribution across all words in the vocabulary. 
#That means that we need to turn the output element from a single integer into a one hot encoding 
#with a 0 for every word in the vocabulary and a 1 for the actual word that the value. 
#Keras provides the to_categorical() function 
#that we can use to convert the integer to a one hot encoding while specifying the number of classes as the vocabulary size.

## one hot encode outputs
y = to_categorical(y, num_classes=vocab_size)

#print(y)
#one hot vector eg.
#y[0]=[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#y[1]=[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]


##############################################################################
##############################################################################
##############################################################################
#data preparation






#model
###############################################################################
###############################################################################
###############################################################################

# define model
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))
#print(model.summary())
#vocab size is the size of vocabalary that is 21 in this case
#10 is the dimension of word embedding
#single lstm layer with 50 units
#output has 1 neuron for each word in vocabulary 
#& output uses softmax activation function to ensure the output is normalized to look like a probability.



# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(X, y, epochs=500, verbose=2)

#it is a multi class classification problem
#multi-class classification problem thus adam optimizer with 500 epoch is used

########################################################################################
########################################################################################
########################################################################################


#testing 
def generate_seq(model, tokenizer, input_text, no_of_words):
	in_text, result = input_text, input_text
	# generate a fixed number of words
	for _ in range(no_of_words):
		# encode the text as integer
		inputTextEncoded = tokenizer.texts_to_sequences([in_text])[0]
		inputTextEncoded = array(inputTextEncoded)
		# predict a word in the vocabulary
		yhat = model.predict_classes(inputTextEncoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text, result = out_word, result + ' ' + out_word
	return result



# evaluate
print(generate_seq(model, tokenizer, 'cat', 6))

# generate a sequence from the model
