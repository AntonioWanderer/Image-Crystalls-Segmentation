import cv2, numpy, pandas, openpyxl, os
import matplotlib.pyplot as plt
import Config

def load_im(filename: str = ""):
    img = cv2.imread("content/" + filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def processing(img, gray, filename):
    N = 1
    h, w = gray.shape
    S = h * w
    ret, th = cv2.threshold(gray, Config.thresholdLevel, Config.pixelMax, cv2.THRESH_BINARY)
    cv2.imwrite(f"results/thresold {filename}", th)

    contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, Config.contourColor, 2)
    cv2.imwrite(f"results/cont {filename}", img)

    area = []
    perimeter = []
    PS = []
    scaled = []
    for i in range(len(contours)):
        item = contours[i]
        segmentArea = cv2.contourArea(item)
        segmentPerimeter = cv2.arcLength(item, True)
        if hierarchy[0, i, 2] != -1:
            child = hierarchy[0, i, 2]
            segmentArea -= cv2.contourArea(contours[child])
            segmentPerimeter += cv2.arcLength(contours[child], True)
        if segmentArea > Config.split_level:
            area.append(segmentArea)
            perimeter.append(segmentPerimeter)
            PS.append(segmentPerimeter / segmentArea)
            scaled.append(segmentPerimeter / (segmentArea*Config.scale))
    # PS = sorted(PS, reverse=True)
    avg = sum(PS) / len(PS)
    df = pandas.DataFrame(data={Config.columns[0]: area,
                                Config.columns[1]: perimeter,
                                Config.columns[2]: PS,
                                Config.columns[3]: scaled})

    return df, avg, N


def graphics(filename, df, avg, N):
    df = df.sort_values(Config.columns[0], ascending=False)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.xlabel("Area")
    plt.ylabel("Perimeter/Area")
    plt.title(filename)
    ax.plot(df[Config.columns[0]].to_list(),df[Config.columns[2]].to_list())
    # ax.scatter(PS)
    ax.plot((0, max(df[Config.columns[0]])), (avg, avg), label=f"Average %.2f 1/px" % avg)
    ax.legend()
    fig.savefig(f'graphs/graph {filename}.jpg')  # save the figure to file
    plt.close(fig)


def createTable(name, df):
    res = df.sort_values(Config.columns[0], ascending=False)
    res.to_excel("tables/" + name, sheet_name='openCV')
