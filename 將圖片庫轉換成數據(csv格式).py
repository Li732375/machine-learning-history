##將圖片庫轉換成數據(csv 格式)
##介面完成
##測試完成
##倘若圖片太大張，excel會打不開喔
##首欄位是檔名，其次就是圖片數據

import os, csv
from PIL import Image 
import numpy as np


###main
##讀取目錄路徑
#'C:\\Users\\user\\Pictures\\Screenshots'
pictrueInputDir = str(input('輸入圖片庫目錄，可以直接拖曳至此：'))
if not os.path.isdir(pictrueInputDir):
    print('路徑驗證異常，請確認路徑')
    
pictrueOutputDir = str(input('輸出 csv 的位置，可以直接拖曳至此，預設輸出在目標的目錄：'))
    
if len(pictrueOutputDir) == 0:
    pictrueOutputDir = pictrueInputDir.split("\\")[:-1]
    pictrueOutputDir = "\\".join(pictrueOutputDir)

if not os.path.isdir(pictrueOutputDir):
    os.mkdir(pictrueOutputDir)
    print('目錄尚未找到，自動建立新目錄 => ', pictrueOutputDir)

pictureFileExtension = str(input('輸入一個指定的副檔名：'))

print('\n開始轉換...\n')
##逐一開啟圖片並轉換

with open( pictrueOutputDir + os.sep + 'imageDataset.csv', 'w', newline = '') as f:

    #確保已寫過欄位名稱
    writed = False
    
    # 逐一查詢檔案清單
    for item in os.listdir(pictrueInputDir):

        target = os.path.join(pictrueInputDir, item)
    
        if os.path.isdir(target):
        #使用 isdir 檢查是否為目錄
        #使用 join 的方式把路徑與檔案名稱串起來
            print('發現目錄: ', item)    
    
        elif os.path.isfile(target):
        #使用isfile判斷是否為檔案
            print('發現檔案: ', item)

            if os.path.splitext(item)[-1] == '.' + pictureFileExtension:
                pic = Image.open(target)
            
                ##在numpy中的陣列，第一個維度是 Y 軸，第二維度是 X 軸。一般而言，
                ##我們說這個影像是 1024 * 768，代表影像的寬度是1024，高度是768；
                ##可是我們說這個矩陣是1024 * 768，卻代表矩陣的高度是1024，寬度是
                ## 768，要特別留意。
                ## Img.getpixel((x,y)) = data[y,x]
                ##影像(寬*高) = 陣列(高(y)*寬(x))
            
                #這裡輸出的是單一張圖片的二維陣列
                pic_arr = np.array(pic)
                print('轉換後維度', pic_arr.shape)

                #列的內容
                row_data = [item]
                
                #降成一維儲存
                pic_arr_One = pic_arr.flatten()
                row_data.extend(pic_arr_One)
                #print(pic_arr_One.shape)

                pictureSize = len(pic_arr_One)

                if writed == False:    
                    #欄位名稱
                    column_name = ['name']
                    column_name.extend(['pixel %d'%i for i in range(pictureSize)])

                    #寫入欄位名稱
                    writer = csv.writer(f)
                    writer.writerow(column_name)

                    writed = True
                    
                #寫入列
                writer.writerow(row_data)
                
            print(item + ' 轉換數據完成\n')
             
        else:
            print('\nerror !!\n')
            os.system('pause')

print('作業完成\n')
os.system('pause')


