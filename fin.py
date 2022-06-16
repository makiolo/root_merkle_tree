def df2zc(df, dcf=1.0, convention='LINEAR'):
    if convention == 'LINEAR':
        zc = (1.0 / df - 1.0) * (1.0 / dcf)
    elif convention == 'EXPONENTIAL':
        zc = -np.log(df) * (1.0 / dcf)
    elif convention == 'YIELD':
        zc = ((1.0 / df) ** (1.0 / dcf)) - 1.0
    else:
        raise Exception('Invalid convection: {}'.format(convention))
    return zc


def zc2df(zc, dcf=1.0, convention='LINEAR'):
    if convention == 'LINEAR':
        df = 1.0 / (1.0 + zc * dcf)
    elif convention == 'YIELD':
        df = 1.0 / ((1.0 + zc) ** dcf)
    elif convention == 'EXPONENTIAL':
        df = np.exp(-zc * dcf)
    else:
        raise Exception('Invalid convection: {}'.format(convention))
    return df
  
