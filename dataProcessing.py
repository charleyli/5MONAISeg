from monai.transforms import LoadImage

# 实例化 transform
loader = LoadImage(image_only=True)

# 读取nii.gz文件
image1 = loader("./data/AbdomenAtlas/BDMAP_00000001/ct.nii.gz")
image2 = loader("./data/AbdomenAtlas/BDMAP_00000001/segmentations/aorta.nii.gz")
image3 = loader("./data/AbdomenAtlas/BDMAP_00000001/segmentations/gall_bladder.nii.gz")

print("类型:", type(image1))
print("维度:", image1.shape)
print("数值范围:", image1.min(), "~", image1.max())
print(image1)
print("类型:", type(image2))
print("维度:", image2.shape)
print("数值范围:", image2.min(), "~", image2.max())

print(image2)


print("类型:", type(image3))
print("维度:", image3.shape)
print("数值范围:", image3.min(), "~", image3.max())