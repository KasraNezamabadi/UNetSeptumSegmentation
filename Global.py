def log(text, have_line=True):
    if have_line:
        print('-'*30)
        print(text)
        print('-'*30)
    else:
        print(text)