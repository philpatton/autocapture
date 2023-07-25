"""Add misidentification errors to a true capture history.

Errors only occur on recaptures. Three types of errors are possible: false 
accepts, mark changes, and ghosts. False accepts result from confusing one 
individual with another. These errors are more likely to happen to occur between
similar animals. Similarity scores are simulated with a right-skewed Beta.

Mark changes mistakenly split a capture history in two, simulating a case where 
the animal changed during the study. Ghosts refer to accidentally attributing a 
single recapture to a new individual, simulating a one-off error. Ghosts and 
mark changes are two forms of false rejects.

Typical usage example:

    alpha = 0.025; beta = 0.025; gamma = 0.01
    mi = MissID(alpha, beta, gamma)
    
    true_history = np.random.default_rng().binomial(1, 0.5, (10, 5))
    capture_history = mi.simulate_capture_history(true_history)
"""

import numpy as np
from src.utils import softmax, first_nonzero

class MissID:
    """Muddies capture histories with misidentification errors. 
    
    Attributes:
        alpha: the proportion of recaptures resulting in ghosts 
        beta: the proportion of recatpures resulting in mark changes
        gamma: the proportion of recaptures resulting in false accepts 
        seed: integer seed for the rng
        rng: np.random.Generator used by the model 
        A: alpha parameter in beta distribution of similarity scores
        B: beta parameter in beta distribution of similarity scores
    """

    def __init__(self, alpha: float, beta: float, gamma: float, 
                 seed: int = None, A: float = 2, B: float = 5):
        """Init the data generator with hyperparameters and init the rng"""
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.A = A
        self.B = B

    def simulate_capture_history(self, true_history: np.ndarray) -> np.ndarray:
        """Simulates misidentification errors for a capture history.
        
        The misidentification errors can be false rejects, with or without mark
        changes, or false accepts.

        Args:
            true_history: N by T matrix indicating capture
        Returns:
            N by T matrix indicating capture, including misid errors 
        """
        capture_history = true_history.copy()

        # both types of misids only occur on recaptures
        recapture_history = self.create_recapture_history(capture_history)

        # flag the recaptures for each error type (false accept or reject)
        flag_dict = self.flag_errors(recapture_history)

        if any(flag_dict['false_accept']):

            capture_history = self.false_accept_process(
                recapture_history=recapture_history,
                flag_dict=flag_dict,
                capture_history=capture_history
            )

        if any(flag_dict['mark_change']):

            capture_history = self.mark_change_process(
                recapture_history=recapture_history,
                flag_dict=flag_dict,
                capture_history=capture_history
            )

        if any(flag_dict['ghost']):

            capture_history = self.ghost_process(
                recapture_history=recapture_history,
                flag_dict=flag_dict,
                capture_history=capture_history
            )

        return capture_history

    def create_recapture_history(self, capture_history: np.ndarray
                                 ) -> np.ndarray:
        """Sets initial captures in capture_history to zero"""

        # zero out the first capture occasion to get the recapture history 
        recapture_history = capture_history.copy()
        capture_count = capture_history.shape[0]
        first_capture_occasion = first_nonzero(capture_history)
        recapture_history[np.arange(capture_count), first_capture_occasion] = 0

        return recapture_history

    def flag_errors(self, recapture_history: np.ndarray) -> dict:
        """Randomly flag recaptures as ghosts, mark changes, false accepts."""

        # draw random uniform for classifying each recapture
        recapture_count = recapture_history.sum()
        dummy_randoms = self.rng.uniform(size=recapture_count)

        # cutoffs for errors 
        abg = self.alpha + self.beta + self.gamma
        bg = self.beta + self.gamma 
        
        # classify errors
        ghost_flag = (abg > dummy_randoms) & (dummy_randoms > bg)
        mark_change_flag = (bg > dummy_randoms) & (dummy_randoms > self.gamma)
        false_accept_flag = dummy_randoms < self.gamma
        
        flag_dict = {
            'ghost':ghost_flag,
            'mark_change':mark_change_flag,
            'false_accept':false_accept_flag
        }

        return flag_dict

    def false_accept_process(self, recapture_history: np.ndarray, 
                             flag_dict: dict, capture_history: np.ndarray
                             ) -> np.ndarray:
        """Move recaptures to other animals. """

        # find the animals for erroneously allocating recaptures    
        false_accept_indices = self.get_error_indices(
            recapture_history,
            flag_dict['false_accept']
        )
        wrong_animals = self.pick_wrong_animals(
            false_accept_indices,
            capture_history
        )
                    
        # copy the recaptures to the misidentified animal 
        capture_history[wrong_animals, false_accept_indices[1]] = 1
                
        # zero out falsely accepted animals
        capture_history[false_accept_indices] = 0

        return capture_history

    def get_error_indices(self, recapture_history, error_flag):
        """Gets the (animal, occasion) of the erroneous recatpure."""
        recapture_idx = recapture_history.nonzero()
        error_animal = recapture_idx[0][error_flag]
        error_occasion = recapture_idx[1][error_flag]

        return error_animal, error_occasion

    def pick_wrong_animals(self, false_accept_indices, capture_history):
        """Picks animals to be confused with for each false acceptance."""
        false_accept_animal, false_accept_occasion = false_accept_indices

        animal_count = capture_history.shape[0]
        similarity = self.simulate_similarity(animal_count)

        def pick_wrong_animal(a, t):
            """Selects animal to be confused with"""

            # animals with higher similarity are more likely to be picked 
            pi = similarity[a].copy()

            # animals not captured cant be mistaken
            not_yet_captured = capture_history[:,:t].max(axis=1) == 0
            pi[not_yet_captured] = 0
            pi = softmax(pi)

            if all(pi == 0):
                raise ValueError('Unable to pick an animal for false acceptance.')
            wrong_animal = np.argmax(self.rng.multinomial(1, pi))

            return wrong_animal

        # choose mididentified animals based on similarity 
        wrong_animal = [pick_wrong_animal(a, t) for a, t 
                        in zip(false_accept_animal, false_accept_occasion)]
        wrong_animal = np.asarray(wrong_animal)
        
        return wrong_animal

    def simulate_similarity(self, animal_count):
        """Simulates similarity scores between each animal from a beta dist."""
        # similutate similarity 
        similarity = self.rng.beta(self.A, self.B, (animal_count, animal_count))

        # ensure sim[i,j] == sim[j,i], and sim[i,i] = 0
        similarity = np.triu(similarity, 1)
        i_lower = np.tril_indices(animal_count, -1)
        similarity[i_lower] = similarity.T[i_lower]

        return similarity

    def mark_change_process(self, recapture_history, flag_dict, capture_history):
        """Split an animal's capture history in two."""
        # create ghost histories for every changed animal
        mark_change_indices = self.get_error_indices(
            recapture_history,
            flag_dict['mark_change']
        )
        mark_change_history = self.create_ghost_history(
            mark_change_indices,
            T=capture_history.shape[1]
        )

        # copy recaptures from the animals original history to the new one
        mark_change_history = self.copy_recaptures_to_changed_animal(
            mark_change_indices,
            mark_change_history,
            recapture_history
        )

        # zero out recapture and subsequent history for changed animal 
        mc_animals, mc_occasions = mark_change_indices
        for animal, occasion in zip(mc_animals, mc_occasions):
            capture_history[animal, occasion:] = 0

        capture_history = np.vstack((capture_history, mark_change_history))

        return capture_history

    def create_ghost_history(self, ghost_indices, T):
        """Create capture histories for false rejects (ghosts or mark change)"""
        _, ghost_occasion = ghost_indices
        
        # create ghost histories
        total_false_rejects = len(ghost_occasion)
        ghost_history = np.zeros((total_false_rejects, T), dtype=int)

        # 'indices' ensures we select each ghost_occasion in turn 
        indices = np.arange(total_false_rejects)
        ghost_history[indices, ghost_occasion] = 1
            
        return ghost_history

    def copy_recaptures_to_changed_animal(
            self, 
            mark_change_indices, 
            mark_change_history, 
            recapture_history
        ):
        """Copies recaptures after the mark change to the ghost history."""
        mark_change_animal, mark_change_occasion = mark_change_indices

        # filling in the ghost history after each change
        for i in range(len(mark_change_animal)):

            # capture history after the mark change
            next_occasion = (mark_change_occasion[i] + 1)
            post_change = recapture_history[mark_change_animal[i], 
                                            next_occasion:]

            # add this history to the ghost history
            mark_change_history[i, next_occasion:] = post_change

        return mark_change_history

    def ghost_process(self, recapture_history, flag_dict, capture_history):
        """Add a recapture to a a new fake history."""
        # create ghost histories for non-mark-changes
        ghost_indices = self.get_error_indices(
            recapture_history,
            flag_dict['ghost']
        )
        ghost_history = self.create_ghost_history(
            ghost_indices, 
            T=capture_history.shape[1]
        )
    
        # for ghosts, zero out recapture in original capture history 
        capture_history[ghost_indices] = 0

        capture_history = np.vstack((capture_history, ghost_history))

        return capture_history