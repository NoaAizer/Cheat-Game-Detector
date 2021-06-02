#!/usr/bin/env python


from pathlib import Path
import xml.etree.ElementTree as ET

DATA_PATH = "data/vs_dumb_bot/eng/"

player_names = []
players = []

def getPlayers(file):
    tree = ET.parse(file)
    return tree.getroot()[0]

def get_games(file ):
    tree = ET.parse(file)
    return tree.getroot()[1:]


def main():
    bot_won = 0
    human_won = 0
    for summary_filename in Path(DATA_PATH).glob('*/Summary.Final.xml'):
        description_path = Path(Path(summary_filename).parent).glob('*.xml').__next__()
        bot_index = 1
        try:
            players = getPlayers(description_path)
            if players[0].attrib["Age"] == "1" and players[0].attrib["Name"] == "Bob":
                bot_index = 0
            game_attrib = get_games(description_path )
            for game in game_attrib:
                h_won = 0
                b_won = 0
                if(game.attrib["PlayerWonIndex"] == str(bot_index)):
                    bot_won+=1
                    b_won += 1
                else:
                    human_won += 1
                    h_won +=1
                print( players[bot_index-1].attrib["Name"] , " won: ", h_won , "lost ", b_won)
        except:
            print ("FAIILLL " + str(summary_filename))
          #  print(player_attrib)
       # copyfile(audio_filename, DeceptionDB_path + "claim_" +  str(file_counter) + ".wav")

    print("bot won %d games." , bot_won)
    print("human won %d games." , human_won)
    #print_stats()

if __name__ == "__main__":
    main()
