import os
from PIL import Image

def split_image_grid(image_path, output_dir, rows=11, cols=5):
    """
    将拼接图像按网格分割成单张小图（无损清晰度）并逆时针旋转90度

    参数:
        image_path: 拼接大图路径
        output_dir: 输出目录
        rows: 行数（默认5）
        cols: 列数（默认11）
    """
    # 打开图像
    img = Image.open(image_path)
    width, height = img.size
    print(f"原始图像尺寸: {width} x {height}")

    # 每个子图的宽高
    tile_width = width // cols
    tile_height = height // rows

    print(f"每个小图尺寸: {tile_width} x {tile_height}")

    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 分割并保存每一张小图
    for row in range(rows):
        for col in range(cols):
            left = col * tile_width
            upper = row * tile_height
            right = left + tile_width
            lower = upper + tile_height

            # 裁剪子图并逆时针旋转90度
            tile = img.crop((left, upper, right, lower))
            tile = tile.rotate(0, expand=True)  # 逆时针旋转90度

            # 命名方式: row{行号}_col{列号}.png
            filename = f"row{row+1}_col{col+1}.png"
            tile.save(os.path.join(output_dir, filename), quality=100)

    print(f"✅ 已成功分割并旋转 {rows * cols} 张图片，保存在：{output_dir}")
# 示例调用
if __name__ == "__main__":
    split_image_grid(
        image_path="img/image.png",  # 拼接图像路径
        output_dir="output_images_ffhq",             # 输出目录
        rows=11,
        cols=5
    )
