def callback():
    print("callback")
    return None

def caller(x):
    return print("caller:", x)

# Cas 1
caller(callback())      # injection du return dans l'appel

# Cas 2
callback()
caller(None)            # injection explicite, mais séparée
