# :#/ GSLF
# Gioele SL Fierro
# 2021
#
# UTILITIES

from os import system, name 

def clear(): 
    ''' Clear the CLI
    '''
  
    # Windows clear
    if name == 'nt': 
        _ = system('cls') 
  
    # *nix clear
    else: 
        _ = system('clear')