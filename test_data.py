import pickle
from subprocess import call

data = pickle.load(open("data/Hanabi-Full_2_6_150.pkl", "rb"))
for agent in data:
	print("AGENT")
	for game in agent:
		print("GAME")
		print(game)
		print(data[agent][game])
		#print(game[0])
		#print(game[1])

