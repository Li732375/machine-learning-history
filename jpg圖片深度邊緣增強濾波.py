## jpg 圖片深度邊緣增強濾波
##介面完成
##測試完成

from os import sep, path, system, mkdir
from PIL import Image, ImageFilter 


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
    os.mkdir(pictrueOutputDir)
    print('目錄尚未找到，自動建立新目錄 => ', pictrueOutputDir)

many = input('重複執行次數?')

if len(many) == 0:
    many = 1
else:
    many = int(many)
print('\n開始轉換...\n')
##開啟圖片並轉換

pic = Image.open(pictrueInputDir)

for i in range(many):
    pic = pic.filter(ImageFilter.DETAIL)

pic.save(pictrueOutputDir + sep + 'deepedge-'
            + pictrueInputDir.split(sep)[-1].split('.')[0] + '.png')

print('作業完成\n')
system('pause')


