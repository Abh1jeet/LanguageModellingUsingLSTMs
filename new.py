from itertools import islice
with open("movies.txt/movies.txt") as myfile:
    head = list(islice(myfile, 10))
print(head)