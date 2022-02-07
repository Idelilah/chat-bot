import discord
from chatbot import predict_and_respond


client = discord.Client()


@client.event
async def on_message(message):
	if message.author ==  client.user:
		return
	if message.content.startswith("$NLPbot"):
		response = predict_and_respond(message.content[8:])
		print(message.content[8:])
		await message.channel.send(response)
print("It's running")
client.run(Token)
