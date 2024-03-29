import numpy as np

def softmax(x):
    return x / x.sum()

def first_nonzero(arr, axis=1, invalid_val=-1):
    """Finds the first nonzero value along an axis."""
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def read_inp(path):
    """Read in a .inp file, standard Mark file. Many TODO items"""
    inp_comment_char = '/*'

    histories = []
    history_counts = []
    with open(path) as f:
        for line in f:
            line = line.partition(inp_comment_char)[0]
            line = line.rstrip()

            split_line = line.split(' ')

            capture_history = split_line[0]
            if capture_history != '':
                histories.append(capture_history)

            history_count = split_line[-1].replace(';','')
            if history_count != '':
                history_counts.append(int(history_count))

                
    # explode the character string into a list 
    exploded = [[*hist] for hist in histories]
    capture_array = np.array(exploded).astype(int)

    history_counts = np.array(history_counts)
    
    # ingnore loss_on_capture
    loss_on_capture = history_counts < 0
    history_counts = np.where(loss_on_capture, -history_counts, history_counts)
    
    return capture_array, history_counts

def summarize_condensed_history(history, history_counts):
            
    individual_history = np.repeat(history, history_counts, axis=0)
    
    result = summarize_individual_history(individual_history)
    
    return result

def summarize_individual_history(history):
    """Convert capture histories into summary statistics"""
    occasion_count = history.shape[1]
    
    # n: total number of animals captured at occasion t 
    n = history.sum(axis=0)

    # summary statistics from the capture histories
    u = []
    m = []
    r = []
    z = []
    for occasion in range(occasion_count):
        
        # characterize each capture 
        captured_this_time = history[:, occasion] == 1
        captured_prior = (history[:, :occasion] > 0).any(axis=1)
        captured_later = (history[:, (occasion + 1):] > 0).any(axis=1)

        # populate the statistics based on each condition 
        uu = history[captured_this_time & ~captured_prior].shape[0]
        mm = history[captured_this_time & captured_prior].shape[0]
        rr = history[captured_this_time & captured_later].shape[0]
        zz = history[~captured_this_time & captured_prior & 
                     captured_later].shape[0]
        
        u.append(uu)
        m.append(mm)
        r.append(rr)
        z.append(zz)
    
    # convert the lists to numpy arrays 
    m = np.array(m)
    u = np.array(u)
    r = np.array(r)
    z = np.array(z)
    
    never_recaptured = n - r  
    m_array = create_m_array(history)
        
    result = {'number_released':n, 'm':m, 'u':u, 'r':r, 'z':z, 
              'never_recaptured':never_recaptured, 'm_array':m_array}
    
    return result

def create_m_array(history):
    """Calculate the so-called m-arrary from an individual capture history.
    
    The m-array is an upper triangle matrix where each cell, m_{i,j}, denotes 
    the number of individuals released at occasion t_i and next encountered 
    alive at occasion t_j.    
    
    Args:
        history: Matrix of shape (n_animals_captured, n_occasions) where 1 
          indicates that the animal was captured at occasion
    """
    _, occasion_count = history.shape
    interval_count = occasion_count - 1

    M_array = np.zeros((interval_count, interval_count), int)
    for occasion in range(occasion_count - 1):

        # which individuals, captured at t, were later recaptured?
        captured_this_time = history[:, occasion] == 1
        captured_later = (history[:, (occasion + 1):] > 0).any(axis=1)
        now_and_later = captured_this_time & captured_later
        
        # when were they next recaptured? 
        remaining_history = history[now_and_later, (occasion + 1):]
        next_capture_occasion = (remaining_history.argmax(axis=1)) + occasion 

        # how many of them were there?
        ind, count = np.unique(next_capture_occasion, return_counts=True)
        M_array[occasion, ind] = count
        
    return M_array

def create_full_array(history):
    
    number_released = history.sum(axis=0)
    number_released = number_released[:-1]
    
    m_array = create_m_array(history)
    never_recaptured = number_released - m_array.sum(axis=1)

    # combine the probabilities into array
    interval_count, _ = m_array.shape
    never_recaptured = np.reshape(never_recaptured, (interval_count, 1))
    full_array = np.hstack((m_array, never_recaptured))

    return full_array

def expit(x):
    return 1 / (1 + np.exp(-x))

def freeman_tukey(observed, expected) -> float:
    '''Calculate the Freeman '''
    D = np.power(np.sqrt(observed) - np.sqrt(expected), 2).sum()
    return D

def fill_lower_diag_ones(x: np.ndarray) -> np.ndarray:
    '''Utility function to set the lower diag to one'''
    return np.triu(x) + np.tril(np.ones_like(x), k=-1)

def bayesian_p_value(replicate, observed) -> float:
    return (replicate >= observed).mean()
