# 1.基本

## 默认项目文件介绍：

~~~
mysite
	manage.py	//项目管理，启动项目，创建app，数据管理，无需修改
    mysite
    	wsgi.py	//
      	asgi.py	//接受网络请求，无需修改
      	__init__.py	//
      	settings.py	//项目配置文件
      	urls.py	//URL 和函数的对应关系
~~~

## APP

~~~
- 项目
	- app：用户管理【表结构，函数。HTML模板，CSS】
	- app：订单管理
	- app：后台管理
	- app：网站
	- app：API
~~~

~~~
vnev/Scripts/activate
manage.py startapp appp01	//创建APP
//app01文件夹中有：
    │  admin.py	//固定-默认后台管理
    │  apps.py	//固定的，app启动类
    │  models.py	//！对数据库操作
    │  tests.py	//固定
    │  views.py//！函数
    │  __init__.py
    │
    └─migrations	//固定-数据库变更记录
            __init__.py
~~~

## 快速开始创建项目

- 首先开始之前确保创建的app已经注册：在settings中写INSTALLED_APPS ----'app01.apps.APP01.config'

- 编写URL和视图函数的对应关系：

  ~~~
  from app01 import views
  urlpatterns = [
      # path('admin/', admin.site.urls),
      
      # www.xxx.com/index/ ---> 函数
      path('index/', views.index),
  ]
  ~~~

- 编写views函数

  ~~~
  from django.shortcuts import render, HttpResponse
  def index(request):
      return HttpResponse("欢迎使用！")
  ~~~

- 启动django项目

  - 命令行

    ```
    python mannage.py runserver
    ```

  - Pycharm直接运行、停止

    