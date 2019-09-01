import random
data = open('data.txt').read()


x=data.split(' ')
idx=0
vocab = {}
for word in x:
	if word not in vocab:
		vocab[idx] = word
		idx=idx+1
	



for i in range(10):
  randomIdx=random.randint(0,len(vocab))
  print(vocab[randomIdx]),