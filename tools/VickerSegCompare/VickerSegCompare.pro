QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    BoundingBox.cpp \
    LeeGraphicsView.cpp \
    batchtestdlg.cpp \
    imgproalg.cpp \
    main.cpp \
    mainwindow.cpp \
    utils.cpp

HEADERS += \
    BoundingBox.h \
    LeeGraphicsView.h \
    batchtestdlg.h \
    imgproalg.h \
    mainwindow.h \
    utils.h

FORMS += \
    batchtestdlg.ui \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

# include opencv
#win32:CONFIG(release, debug|release): LIBS += -LC:/Applications/opencv/ManualMake/x86/vc16/lib/ -lopencv_world440
#else:win32:CONFIG(debug, debug|release): LIBS += -LC:/Applications/opencv/ManualMake/x86/vc16/lib/ -lopencv_world440d
#win32:CONFIG(release, debug|release): LIBS += -LC:/Applications/opencv/OfficialInstall/opencv/build/x64/vc15/lib -lopencv_world460
#else:win32:CONFIG(debug, debug|release): LIBS += -LC:/Applications/opencv/OfficialInstall/opencv/build/x64/vc15/lib -lopencv_world460d
win32:CONFIG(release, debug|release): LIBS += -LC:/Applications/Opencv/OfficialInstall/opencv/build/x64/vc15/lib -lopencv_world460
else:win32:CONFIG(debug, debug|release): LIBS += -LC:/Applications/Opencv/OfficialInstall/opencv/build/x64/vc15/lib -lopencv_world460d

#INCLUDEPATH += C:/Applications/opencv/ManualMake/include
#DEPENDPATH += C:/Applications/opencv/ManualMake/include
#INCLUDEPATH += C:/Applications/opencv/OfficialInstall/opencv/build/include
#DEPENDPATH += C:/Applications/opencv/OfficialInstall/opencv/build/include
INCLUDEPATH += C:/Applications/Opencv/OfficialInstall/opencv/build/include
DEPENDPATH += C:/Applications/Opencv/OfficialInstall/opencv/build/include
