## jpg 圖片亮度提升
##介面完成
##測試完成

from os import sep, path, system, mkdir
from PIL import Image, ImageEnhance 


###main
##讀取目錄路徑
pictrueInputDir = str(input('輸入圖片，可以直接拖曳至此：'))
if not path.isfile(pictrueInputDir):
    print('路徑驗證異常，請確認路徑')
    
pictrueOutputDir = str(input('輸出位置，可以直接拖曳至此，預設輸出在目標的目錄：'))
    
if len(pictrueOutputDir) == 0:
    pictrueOutputDir = pictrueInputDir.split("\\")[:-1]
    pictrueOutputDir = "\\".join(pictrueOutputDir)

if not path.isdir(pictrueOutputDir):
    mkdir(pictrueOutputDir)
    print('目錄尚未找到，自動建立新目錄 => ', pictrueOutputDir)

fac = str(input('增強值(單一值，預設不調整)?'))

if len(fac) == 0:
    fac = 1
else:
    fac = int(fac)
    
many = input('重複執行次數?')

if len(many) == 0:
    many = 1
else:
    many = int(many)
    
print('\n開始轉換...\n')
##開啟圖片並轉換

pic = Image.open(pictrueInputDir)

for i in range(many):
    pic = ImageEnhance.Brightness(pic).enhance(fac)

pic.save(pictrueOutputDir + sep + 'l' + str(fac) + '-'
            + pictrueInputDir.split(sep)[-1].split('.')[0] + '.png')

print('作業完成\n')
system('pause')


