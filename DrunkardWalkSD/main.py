from time import time                       # For file R&W operations
import numpy as np                          # General numerical necessities
from matplotlib import pyplot as plt        # For plotting
from typing import List                     # Typing hints

class Drunkard: 
    def __init__(self, sidewalk_size: int) -> None:
        """Creates a new drunkard (walker) with a given probability of stepping right.

        Args:
            sidewalk_size (int): Size of the sidewalk that the walker is placed.
        """
        # Position of the walker -- the walker starts at the middle of the sidewalk
        self.pos = int(sidewalk_size / 2.0)
        # Probability of walking one step to the right
        self.coin_p = None # Dummy value -- it's get updated each time before walk()

    def walk(self) -> int:
        """Flips a coin and moves the drunkard one step."""
    
        if np.random.uniform() <= self.coin_p:
            self.set_pos(self.pos + 1)
            
        else: 
            self.set_pos(self.pos - 1)
            
        return self.get_pos()
    
    # Position 
    def set_pos(self, new_pos: int) -> None:
        self.pos = new_pos

    def get_pos(self) -> int:
        return self.pos

    # Coin
    def set_coin_p(self, new_coin_p: float) -> None:
        self.coin_p = new_coin_p

    def get_coin_p(self) -> float:
        return self.coin_p
    
class Sidewalk:
    """Environment of a single drunkard -- can be interpreted as the original number line."""

    def __init__(self, size: int, coins: List) -> None:
        """Creates a sidewalk (number line) for the drunkard to walk on.

        Args:
            size (int): Length of the sidewalk.
            coin_p (float, optional): Coin probability of stepping right. Defaults to 0.50 (fair coin).
        """
        # Each sidewalk automatically has a drunkard attached to it
        self.drunkard = Drunkard(size)
        self.size = size
        # Stores each position of the drunkard over time
        self.wandering_pos = []
        # Stores the arrays of coins for each slot of the sidewalk
        self.coins = coins

    def wander(self, end_step: int = 1_000) -> List[int]:
        """Simulate the drunkard's walk over a number of steps."""
        
        # Walks the drunkard for as many steps as specified in the method call
        for _ in range(end_step):
            # Selects the coin assigned to the current position occupied by the walker
            new_coin = self.coins[self.drunkard.get_pos()]
            # Updates the walker's coin 
            self.drunkard.set_coin_p(new_coin)
            # Each step is recorded in the "wandering_pos" attribute
            self.wandering_pos.append(self.drunkard.walk())
            
        # The positions traveled by the walker are returned for statistical analysis
        return self.wandering_pos
        
    # Auxiliar method -- gets the sidewalk's size
    def get_size(self) -> int:
        return self.size
    
class City:
    """Environment for multiple sidewalks -- runs many simulations for statistical analysis."""

    def __init__(self, n_sidewalks: int, sidewalk_size: int, coin_W: float) -> None:
        # Number of sidewalks in the city
        self.n_sidewalks = n_sidewalks
        # Size of the city's sidewalks 
        self.sidewalk_size = sidewalk_size
        # Amplitude of possible coins prob. value -- the higher the amplitude the wider 
        # the interval between the lowest and highest possible walk-right-probability values.
        self.coin_W = coin_W    # ranges from 0 to 1

        # DISCLAIMER: a set of sidewalks is called a "pub" (yes, like the ones in Britain); 
        # The following lists are "lists of lists": they contain data from each sidewalk simulated in
        # the city, or derived from them; 
        
        # Stores the lists of positions from each sidewalk simulated
        self.pub_positions = []
        # Stores the average of every walker's position in a given time (ie. record every walker's position 
        # at step 1, 2, ... and average them, then store in this list)
        self.pub_average = []
        # Stores the dispersion (STD) of every walker in a given time -- same logic as above
        self.pub_std = []

    def generate_coins(self) -> List[float]:
        low_b = ((-1.0 * self.coin_W) / 2.0) + 0.5  
        high_b = (self.coin_W / 2.0) + 0.5         
        coins = []
        
        for _ in range(self.sidewalk_size):    
            new_coin = np.random.uniform(low_b, high_b)
            coins.append(new_coin)
            
        return coins
        
    def roam(self, end_step: int = 500) -> List[List[int]]:
        """Executes each sidewalk simulation in succession and stores the positions for 
        statistical analysis. 

        Args:
            end_step (int, optional): The maximum number of steps that the walkers
            are going to traverse. Defaults to 500.

        Returns:
            List[List[int]]: List containing lists of the positions of each walker over time.
        """
        
        # Executes the random walk for each sidewalk specified in the City's initialization
        for _ in range(self.n_sidewalks):
            # Creates a new sidewalk with a given size and generates the necessary coins
            sidewalk = Sidewalk(self.sidewalk_size, self.generate_coins())
            # Executes the random walk for as many steps as needed
            positions = sidewalk.wander(end_step)
            # Stores the generated array
            self.pub_positions.append(positions)
            
        # Returns the list of lists
        return self.pub_positions

    def calc_pub_avg(self) -> List[float] | None:
        """Calculates and returns the average position over time across all sidewalks.

        Returns:
            List[float] | None: List of average positions over time
        """
        
        # Prevents user shenanigans: trying to calc average of nothing
        if not all(isinstance(row, (list, np.ndarray)) for row in self.pub_positions):
            return None

        # Transforms the pub_positions array into a numpy one (a matrix if you will) 
        # for ease of manipulation
        self.pub_positions = np.array(self.pub_positions, dtype=float)
        # Empties the pub_average array to prevent computational error if the method
        # has been run previously
        self.pub_average = []

        # Traverses each column to calculate its average -- each column of the matrix
        # is a given step: the ith column is the ith step.
        for step in range(self.pub_positions.shape[1]):
            # Selects the desired column from the matrix
            column = self.pub_positions[:, step]
            # Calculates its average and stores in the array
            self.pub_average.append(np.average(column))

        # Returns the list of averages
        return self.pub_average

    def calc_pub_std(self) -> List[float] | None:
        """Calculates and returns the dispersion (STD) over time across all sidewalks.

        Returns:
            List[float] | None: List of dispersions over time
        """
        
        # Prevents user shenanigans: trying to calc STD of nothing
        if not all(isinstance(row, (list, np.ndarray)) for row in self.pub_positions):
            return None

        # Transforms the pub_positions array into a numpy one (a matrix if you will) 
        # for ease of manipulation
        self.pub_positions = np.array(self.pub_positions, dtype=float)
        # Empties the pub_std array to prevent computational error if the method
        # has been run previously
        self.pub_std = []

        # Traverses each column to calculate its dispersion -- each column of the matrix
        # is a given step: the ith column is the ith step.
        for step in range(self.pub_positions.shape[1]):
            self.pub_std.append(np.std(self.pub_positions[:, step]))

        # Returns the list of dispersions
        return self.pub_std

    def make_avg_graph(self) -> None:
        """Plots the averages over time of the random walks. 
        """
        # Title -- Changes automatically with the number of sidewalks
        plt.title(f"Average Position in Time for {self.n_sidewalks} Drunkards")
        plt.xlabel("Time (Steps)")
        plt.ylabel("Average Position")
        
        plt.plot(self.calc_pub_avg())
        
        # Saves the plot -- filename automatically configured for timestamp, coin, number of
        # sidewalks and their size
        plt.savefig(
            f"AvgPos_{time()}_"
            f"nsw={self.n_sidewalks}_"
            f"sws={self.sidewalk_size}_.png"
        )
        plt.close()

    def make_std_graph(self) -> None:
        """Plots the dispersion over time of the random walks.
        """
        # Title -- Changes automatically with the number of sidewalks
        plt.title(f"Dispersion for {self.n_sidewalks} Drunkards")
        plt.xlabel("Time (Steps)")
        plt.ylabel("Dispersion / Standard Deviation")

        plt.plot(self.calc_pub_std())

        # Saves the plot -- filename automatically configured for timestamp, coin, number of
        # sidewalks and their size
        plt.savefig(
            f"Disp_{time()}_"
            f"nsw={self.n_sidewalks}_"
            f"sws={self.sidewalk_size}_.png"
        )
        
        # Closes the plot to prevent memory accumulation and plotting over the same plot
        plt.close()