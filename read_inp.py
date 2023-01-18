import numpy as np

def main():

    # read in the inp file 
    path = 'capsid.txt'
    history, history_counts = read_inp(path)
    
    summary = summarize_condensed_history(history, history_counts)
    
    return summary

def read_inp(path):
    
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

                
    # explode the character string 0011 into a list 
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
    
    occasion_count = history.shape[1]
    
    # n: total number of animals captured at occasion t 
    n = history.sum(axis=0)

    u = []
    m = []
    r = []
    z = []
    for occasion in range(occasion_count):
        
        captured_this_time = history[:, occasion] == 1
        captured_prior = (history[:, :occasion] > 0).any(axis=1)
        captured_later = (history[:, (occasion + 1):] > 0).any(axis=1)

        uu = history[captured_this_time & ~captured_prior].shape[0]
        mm = history[captured_this_time & captured_prior].shape[0]
        rr = history[captured_this_time & captured_later].shape[0]
        zz = history[~captured_this_time & captured_prior & 
                     captured_later].shape[0]
        
        u.append(uu)
        m.append(mm)
        r.append(rr)
        z.append(zz)
        
    m_array = create_m_array(history)
        
    result = {'n':n, 'm':m, 'u':u, 'r':r, 'z':z, 'm_array':m_array}
    
    return result

def create_m_array(history):
    
    occasion_count = history.shape[1]
    
    M_array = np.zeros((occasion_count - 1, occasion_count))
    for occasion in range(occasion_count - 1):

        captured_this_time = history[:, occasion] == 1
        captured_later = (history[:, (occasion + 1):] > 0).any(axis=1)
        now_and_later = captured_this_time & captured_later
        
        remaining_history = history[now_and_later, (occasion + 1):]
        next_capture_occasion = (remaining_history.argmax(axis=1)) + occasion + 1

        ind, count = np.unique(next_capture_occasion, return_counts=True)

        M_array[occasion, ind] = count
        
    return M_array

res = main()