import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors


# 读取图像为jpg格式
img = cv2.imread('D:\picture.jpg')

# 图片效果展示 RGB颜色空间展示与HSV颜色空间展示
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

r, g, b = cv2.split(img)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
norm = colors.Normalize(vmin=-1., vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()

hsv_nemo = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_nemo)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()

# 图像预处理，得到分割后的图像结果
light_blue = (85, 40, 160)
dark_blue = (120, 225, 225)
mask = cv2.inRange(hsv_nemo, light_blue, dark_blue)
result = cv2.bitwise_and(img, img, mask=mask)
plt.subplot(1, 2, 1)
plt.axis('off')
plt.rcParams['font.sans-serif']=['SimHei']
plt.title('分割掩模')
plt.imshow(mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.axis('off')
plt.rcParams['font.sans-serif']=['SimHei']
plt.title('处理结果')
plt.show()

# 初始化位置数组，a表示纵坐标，b表示横坐标
a = []
b = []
c = [10, 20, 40, 60, 80, 100]

# 鼠标选点记录位置函数
def on_EVENT_LBUTTONDOWN(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        if int(len(a)/2)<len(c):
            m = "%d" % c[int(len(a)/2)]
        a.append(x)
        b.append(y)
        if ((len(a) % 2) == 0) * (len(a) != 0) != 0:
            cv2.line(img, (a[-2],b[-2]), (a[-1],b[-2]), (0, 0, 255), 3, cv2.LINE_AA)
            cv2.line(img, (a[-2],b[-2]), (a[-2],b[-1]), (0, 0, 255), 3, cv2.LINE_AA)
            cv2.line(img, (a[-1],b[-1]), (a[-2],b[-1]), (0, 0, 255), 3, cv2.LINE_AA)
            cv2.line(img, (a[-1],b[-1]), (a[-1],b[-2]), (0, 0, 255), 3, cv2.LINE_AA)
            if int(len(a) / 2) <= len(c):
                cv2.putText(img, m, (x, y), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), thickness=3)
            else:
                cv2.putText(img, "X", (x, y), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), thickness=3)
            print('您已选择'+str(int(len(a)/2))+'个区域')
            cv2.imshow("Please select the target area", img)
        cv2.imshow("Please select the target area", img)

# 显示输入图像，并进行颜色模糊处理，为了能够使选出的点更加均匀
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
plt.subplot(1, 2, 1)
plt.axis('off')
plt.rcParams['font.sans-serif']=['SimHei']
plt.title('原图像')
plt.imshow(img)
img = cv2.GaussianBlur(img, (15,15), 0)
img = cv2.blur(img, (1, 15))
plt.subplot(1, 2, 2)
plt.axis('off')
plt.rcParams['font.sans-serif']=['SimHei']
plt.title('滤波图像')
plt.imshow(img)
plt.show()

mask = mask/255
for i in range(3):
    img[:, :, i] = img[:, :, i] * mask
cv2.namedWindow('Please select the target area', cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Please select the target area", on_EVENT_LBUTTONDOWN)
cv2.imshow("Please select the target area", img)
cv2.waitKey(0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 显示所选择的几个点
print(a, b)

# 由所选位置得到图像对应灰度值大小
pixel = []
pixeltrain = []
for i in range(len(a)):
    if i % 2 == 0:
        x = img[b[i]+5: b[i+1]-5, a[i]+5:a[i+1]-5, :]
        y = np.sum(x[:, :, 1])/np.count_nonzero(x)
        pixel.append(y)
for i in range(len(c)):
    pixeltrain.append(pixel[i])
pixeltest = pixel[-1]

# 构建线性回归模型
c = np.array(c).reshape((-1, 1))
model = LinearRegression()
print(pixeltrain)
model.fit(c, pixeltrain)
r_sq = model.score(c, pixeltrain)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
x_pred = (pixeltest-model.intercept_)/model.coef_
print('predicted c:', x_pred)

# 呈现结果
width = np.linspace(0, 100, 1000)
height = model.coef_*width+model.intercept_
plt.figure(figsize=(10, 10), dpi=100)
plt.scatter(c, pixeltrain)
plt.plot(width, height)
plt.scatter([x_pred], [pixeltest], s=25, c='r')
plt.show()