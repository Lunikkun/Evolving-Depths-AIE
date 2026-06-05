import random
import config

class Director:
    def __init__(self, use_bandit=True):
        self.use_bandit = use_bandit
        self.bandit = None
        self.last_engagement = 0.5
        self.generator = None  # Inizializziamo l'attributo che main.py cerca
        
        if self.use_bandit:
            try:
                from ai.bandit_dea import BanditDirector
                self.bandit = BanditDirector()
            except ImportError:
                print("Warning: ai.bandit_dea.BanditDirector not found.")
                self.use_bandit = False

    def _choose_difficulty(self) -> int:
        if self.use_bandit and self.bandit is not None:
            if hasattr(self.bandit, 'choose_difficulty'):
                return self.bandit.choose_difficulty()
        return 1

    def generate_next_room(self, stats: dict, key_press_rate: float, exits: list = None):
        from ai.generator import RoomGenerator
        
        # Aggiorniamo l'engagement
        self.last_engagement = min(1.0, max(0.0, key_press_rate / 5.0))
        
        # Creiamo l'istanza e la salviamo in self.generator
        self.generator = RoomGenerator() 
        
        difficulty = self._choose_difficulty()
        
        # Creiamo la mappa
        grid = self.generator.create_map(
            difficulty_level=difficulty, 
            engagement=self.last_engagement
        )
        
        attempts = 0 
        
        return grid, difficulty, attempts