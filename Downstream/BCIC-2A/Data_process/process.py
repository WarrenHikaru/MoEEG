import os
import random
import glob
import shutil


def copy_and_rename_sub_files(source_folder, dest_folder, target_sub, all_subs):
    """
    随机选择指定sub文件的30%，复制并将其后缀改为其他sub，保存到指定文件夹

    参数:
    source_folder: 源文件所在文件夹路径
    dest_folder: 目标文件夹路径（修改后的文件将保存到这里）
    target_sub: 要处理的目标sub（如'sub1'）
    all_subs: 所有可能的sub列表（如['sub1', 'sub2', ..., 'sub11']）
    """
    # 确保目标sub在所有sub列表中
    if target_sub not in all_subs:
        print(f"错误: 目标sub '{target_sub}' 不在所有sub列表中")
        return

    # 创建目标文件夹（如果不存在）
    os.makedirs(dest_folder, exist_ok=True)

    # 获取所有目标sub的文件
    pattern = os.path.join(source_folder, f"*.{target_sub}")
    target_files = glob.glob(pattern)

    if not target_files:
        print(f"没有找到后缀为'{target_sub}'的文件")
        return

    print(f"找到 {len(target_files)} 个后缀为'{target_sub}'的文件")

    # 计算需要复制并重命名的文件数量（10%）
    num_to_process = int(len(target_files) * 0.1)
    if num_to_process == 0 and len(target_files) > 0:
        num_to_process = 1  # 至少处理一个文件，如果有文件的话

    print(f"将随机选择 {num_to_process} 个文件进行复制和重命名")

    # 随机选择要处理的文件
    files_to_process = random.sample(target_files, num_to_process)

    # 其他sub的列表（排除目标sub）
    other_subs = [sub for sub in all_subs if sub != target_sub]

    # 复制并命名文件
    processed_count = 0
    for file_path in files_to_process:
        # 获取文件名（不包含路径和后缀）
        file_name = os.path.basename(file_path)
        base_name = file_name.rsplit(f'.{target_sub}', 1)[0]

        # 随机选择一个其他的sub
        new_sub = random.choice(other_subs)

        # 新的文件路径
        new_file_name = f"{base_name}.{new_sub}"
        new_file_path = os.path.join(dest_folder, new_file_name)

        # 检查新文件名是否已存在，如果存在则选择另一个sub
        while os.path.exists(new_file_path):
            new_sub = random.choice(other_subs)
            new_file_name = f"{base_name}.{new_sub}"
            new_file_path = os.path.join(dest_folder, new_file_name)

        # 复制文件
        shutil.copy2(file_path, new_file_path)  # 使用copy2保留文件元数据
        processed_count += 1
        print(f"已处理: {file_name} -> {new_file_name}")

    print(f"操作完成，共处理了 {processed_count} 个文件，保存到 {dest_folder}")


if __name__ == "__main__":
    # 配置参数
    source_folder = "../PhysioNetP300/1/"  # 源文件所在文件夹路径
    dest_folder = "sub2/1/"  # 修改后的文件保存路径
    target_sub = "sub2"  # 要处理的目标sub
    all_subs = ["sub1","sub2","sub3","sub4","sub5","sub6","sub7","sub9","sub11",]  # sub1到sub11

    # 调用函数
    copy_and_rename_sub_files(source_folder, dest_folder, target_sub, all_subs)
