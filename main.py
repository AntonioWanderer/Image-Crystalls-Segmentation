import Config, Methods, os

if __name__ == "__main__":
    for filename in os.listdir("content/"):
        fname = filename[:filename.index(".")]
        img, gray = Methods.load_im(filename)
        df, avg, N = Methods.processing(img, gray, filename)
        Methods.graphics(fname, df, avg, N)
        Methods.createTable(fname+".xlsx", df)
