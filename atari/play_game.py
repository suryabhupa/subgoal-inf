#!/usr/bin/env python
# python_example.py
# Author: Ben Goodrich
#
# This is a direct port to python of the shared library example from
# ALE provided in doc/examples/sharedLibraryInterfaceExample.cpp
import sys
import os
from random import randrange
from ale_python_interface import ALEInterface

if len(sys.argv) < 2:
  print 'Usage:', sys.argv[0], 'rom_file'
  sys.exit()

ale = ALEInterface()

# Get & Set the desired settings
ale.setInt('random_seed', 123)

# Set USE_SDL to true to display the screen. ALE must be compilied
# with SDL enabled for this to work. On OSX, pygame init is used to
# proxy-call SDL_main.
USE_SDL = True
if USE_SDL:
  if sys.platform == 'darwin':
    import pygame
    pygame.init()
    ale.setBool('sound', False) # Sound doesn't work on OSX
  elif sys.platform.startswith('linux'):
    ale.setBool('sound', True)
  ale.setBool('display_screen', True)
  
exp_path =  "records/exp{}".format(sys.argv[2])
os.mkdir(exp_path)
ale.setString("record_screen_dir", exp_path)

# Load the ROM file
ale.loadROM(sys.argv[1])
print sys.argv[1]
#ale.loadROM("../../Roms/ROMS/montezuma_revenge.bin")

# Get the list of legal actions
legal_actions = ale.getLegalActionSet()
one_hot = {0:"1 0 0 0 0 0 0 0 0", 
  2:"0 1 0 0 0 0 0 0 0", 
  3:"0 0 1 0 0 0 0 0 0", 
  4:"0 0 0 1 0 0 0 0 0", 
  5:"0 0 0 0 1 0 0 0 0", 
  6:"0 0 0 0 0 1 0 0 0", 
  7:"0 0 0 0 0 0 1 0 0", 
  8:"0 0 0 0 0 0 0 1 0", 
  9:"0 0 0 0 0 0 0 0 1"}

# Create a file to keep track of actions
f = open(exp_path + "/" + "game_actions.txt", "w")

# Play 10 episodes
for episode in xrange(10):
  total_reward = 0
  while not ale.game_over():
    user_action = raw_input()
    if user_action == '':
      user_action = 0
      f.write(one_hot[user_action] + "\n")
    else:
      try:
        user_action = int(user_action)
        f.write(one_hot[user_action] + "\n")
      except e:
        print "Not valid move '{}'.\n".format(user_action)
    a = legal_actions[user_action]

    # Apply an action and get the resulting reward
    reward = ale.act(a);
    total_reward += reward
  print 'Episode', episode, 'ended with score:', total_reward
  ale.reset_game()

f.close()
