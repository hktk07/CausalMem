from glob import glob
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # ✅ 只保留一个输出目录（去掉 output_dir1 / output_dir2）
    parser.add_argument(
        "--output_dir",
        required=True,
        help="path to evaluation output directory"
    )

    parser.add_argument(
        "--eval-type",
        default="multi_choice",
        help="evaluation type"
    )

    parser.add_argument(
        "--num-chunks",
        type=int,
        default=4
    )

    return parser.parse_args()


def main():
    args = parse_args()

    ori_dir = args.output_dir
    pred_type = args.eval_type

    # 找文件（兼容 chunk / 非 chunk）
    files = glob(f"{ori_dir}/{args.num_chunks}_*")
    if len(files) == 0:
        files = glob(f"{ori_dir}/*")

    if pred_type != "multi_choice":
        raise NotImplementedError("Only multi_choice supported in this version.")

    total_cnt, total_acc = 0, 0
    OCR_cnt = ACR_cnt = ATR_cnt = STU_cnt = FPD_cnt = OJR_cnt = 0
    OCR_acc = ACR_acc = ATR_acc = STU_acc = FPD_acc = OJR_acc = 0

    used_id = set()
    lines = []

    # ========== 读取数据（单目录，不再 cross-dir merge） ==========
    for file in files:
        with open(file, "r") as f:
            for row in f.readlines():
                item = eval(row)
                if item["id"] in used_id:
                    continue
                used_id.add(item["id"])
                lines.append(item)

    # ========== 统计 ==========
    for item in lines:
        total_cnt += 1
        category = item["task"]

        if category == "OCR":
            OCR_cnt += 1
        elif category == "ACR":
            ACR_cnt += 1
        elif category == "ATR":
            ATR_cnt += 1
        elif category == "STU":
            STU_cnt += 1
        elif category == "FPD":
            FPD_cnt += 1
        elif category == "OJR":
            OJR_cnt += 1

        if item["acc"] == "True":
            total_acc += 1
            if category == "OCR":
                OCR_acc += 1
            elif category == "ACR":
                ACR_acc += 1
            elif category == "ATR":
                ATR_acc += 1
            elif category == "STU":
                STU_acc += 1
            elif category == "FPD":
                FPD_acc += 1
            elif category == "OJR":
                OJR_acc += 1

    # ========== 输出 ==========
    def safe_print(name, acc, cnt):
        if cnt > 0:
            print(f"{name} acc: {acc / cnt:.4f}, cnt: {cnt}")

    safe_print("OCR", OCR_acc, OCR_cnt)
    safe_print("ACR", ACR_acc, ACR_cnt)
    safe_print("ATR", ATR_acc, ATR_cnt)
    safe_print("STU", STU_acc, STU_cnt)
    safe_print("FPD", FPD_acc, FPD_cnt)
    safe_print("OJR", OJR_acc, OJR_cnt)

    if total_cnt > 0:
        print(f"\nTOTAL acc: {total_acc / total_cnt:.4f}, total_cnt: {total_cnt}")


if __name__ == "__main__":
    main()