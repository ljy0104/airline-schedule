# -*- coding: utf-8 -*-
"""
Created on 2020/10/7  9:08 

# Author： Jinyu
"""
import os
import pandas as pd
import re

def cvs_to_excel(file_path,cvs_name,excel_name):
    #dir = 'F:\jinyu visit\Pyomo\Pyomo3_smallPDX\总结结果\\1019'
    dir = file_path
    os.chdir(dir)
    writer = pd.ExcelWriter(str(excel_name)+'.xlsx')
    list_csv = os.listdir(dir)  # 列出文件夹下所有的目录与文件
    for name in list_csv:
        if  cvs_name in name:
            data= pd.read_csv(name, encoding="gbk", index_col=0)
            number = re.findall("\d+",name)  # 输出结果为列表
            if number:
                data.to_excel(writer, sheet_name=number[0])
            else:
                data.to_excel(writer, sheet_name='initial')
    writer.save()
name=['assignment','flight','timetable','result']
path='F:\jinyu visit\Dropbox\Pyomo_in_China\Pyomo3_smallPDX\总结结果\\5小时60航班时差调'
for i in name:
    cvs_to_excel(path,i,i+'_compile')
