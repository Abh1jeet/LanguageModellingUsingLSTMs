from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
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



# create line-based sequences
sequences = list()
for i in range(1, len(encoded)):
	sequence = encoded[i-1:i+1]
	sequences.append(sequence)

#print(sequences)
#[[2, 1], [2, 1, 3], [2, 1, 3, 4], [2, 1, 3, 4, 5], [2, 1, 3, 4, 5, 6], [2, 1, 3, 4, 5, 6, 7],
#[8, 9], [8, 9, 10], [8, 9, 10, 11], [8, 9, 10, 11, 12], [8, 9, 10, 11, 12, 13], [2, 14], [2, 14, 15], [2, 14, 15, 1],
#[2, 14, 15, 1, 16], [2, 14, 15, 1, 16, 17], [2, 14, 15, 1, 16, 17, 18], [1, 3], [1, 3, 19], [1, 3, 19, 20], [1, 3, 19, 20, 21]]


#now as all sequences are of different lengths we will pad them to make them all of max length.
# pad input sequences
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')

#print(sequences)
#[[ 0  0  0  0  0  2  1],[ 0  0  0  0  2  1  3],[ 0  0  0  2  1  3  4],[ 0  0  2  1  3  4  5]
#[ 0  2  1  3  4  5  6],[ 2  1  3  4  5  6  7],[ 0  0  0  0  0  8  9],[ 0  0  0  0  8  9 10],[ 0  0  0  8  9 10 11]




# split into input and output elements
sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]
#x me sara except last line 
#y me last wala
y = to_categorical(y, num_classes=vocab_size)
#changing y to onehot vector



#data manipualtion end
###############################################################################
###############################################################################
###############################################################################








#model
###############################################################################
###############################################################################
###############################################################################

# define model
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_length-1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))
#onlyinput size is change to maxlength-1

# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit network
model.fit(X, y, epochs=500, verbose=2)





#model
###############################################################################
###############################################################################
###############################################################################
# generate a sequence from a language model
def generate_seq(model, tokenizer, max_length, input_text, no_of_words):
	in_text = input_text
	# generate a fixed number of words
	for _ in range(no_of_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# pre-pad sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
		# predict probabilities for each word
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
	return in_text


print(generate_seq(model, tokenizer, max_length-1, 'cat', 6))
