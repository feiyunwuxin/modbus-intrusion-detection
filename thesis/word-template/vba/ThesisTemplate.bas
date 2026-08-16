Attribute VB_Name = "ThesisTemplate"
' ============================================================================
'  ThesisTemplate.bas - 硕士论文诊断型模板 VBA 宏
'  配套 LaTeX 模板中的 \pitfall{} / \todo{} / \grade{} / \good{} 宏
'
'  安装方法：
'    1. Word 中按 Alt+F11 打开 VBA 编辑器
'    2. 文件 → 导入文件 → 选择本文件
'    3. Alt+F8 运行宏
'
'  包含宏：
'    InsertPitfall      - 插入红色"学生常见错误"框
'    InsertTodo         - 插入黄色"待补充内容"框
'    InsertGood         - 插入绿色"优秀示例"标记
'    InsertGrade        - 插入蓝色"老师评语"框
'    TogglePitfall      - 显示/隐藏所有诊断框
'    NumberHeadings     - 自动设置章节编号
'    InsertCaptionFig   - 插入图题注
'    InsertCaptionTable - 插入表题注
' ============================================================================

Option Explicit

' 颜色常量
Private Const COLOR_PITFALL_BG As Long = &HEBEBFF   ' 浅红 BGR
Private Const COLOR_PITFALL_BD As Long = &H40C0C0   ' 红色边框
Private Const COLOR_TODO_BG As Long = &HFCFAFF     ' 浅黄
Private Const COLOR_TODO_BD As Long = &H8055FF     ' 橙色
Private Const COLOR_GOOD_TXT As Long = &H408000    ' 绿色
Private Const COLOR_GRADE_BG As Long = &HFFF5EB    ' 浅蓝
Private Const COLOR_GRADE_BD As Long = &HC06040    ' 蓝色

' ============================================================================
'  InsertPitfall - 插入红色"学生常见错误"框（等价 \pitfall{}）
' ============================================================================
Public Sub InsertPitfall()
    Dim title As String
    Dim content As String

    title = InputBox("请输入错误编号（如 P-1.4）：", "学生常见错误", "P-")
    If title = "" Then Exit Sub

    content = InputBox("请输入错误描述：", "学生常见错误", "")
    If content = "" Then Exit Sub

    InsertBox "⚠️ 学生常见错误 [" & title & "]", content, _
              COLOR_PITFALL_BG, COLOR_PITFALL_BD, RGB(192, 0, 0)
End Sub

' ============================================================================
'  InsertTodo - 插入黄色"待补充内容"框（等价 \todo{}）
' ============================================================================
Public Sub InsertTodo()
    Dim owner As String
    Dim content As String

    owner = InputBox("请输入负责人（默认"学生"）：", "待补充内容", "学生")
    If owner = "" Then owner = "学生"

    content = InputBox("请输入待补充内容：", "待补充内容", "")
    If content = "" Then Exit Sub

    InsertBox "📝 待补充（" & owner & "）", content, _
              COLOR_TODO_BG, COLOR_TODO_BD, RGB(192, 96, 0)
End Sub

' ============================================================================
'  InsertGood - 插入绿色"优秀示例"标记（等价 \good{}）
' ============================================================================
Public Sub InsertGood()
    Dim content As String

    content = InputBox("请输入优秀示例说明：", "优秀示例", "")
    If content = "" Then Exit Sub

    Selection.TypeText "[✓ 范例] " & content
    Selection.HomeKey Unit:=wdLine
    Selection.MoveDown Unit:=wdLine, Count:=1

    With Selection.Font
        .Color = RGB(0, 128, 0)
        .Bold = True
    End With

    ' 找到"[✓ 范例]"位置并加粗着色
    Selection.HomeKey Unit:=wdLine
    Selection.Find.Text = "[✓ 范例]"
    Selection.Find.Execute
    If Selection.Find.Found Then
        With Selection.Font
            .Color = RGB(0, 128, 0)
            .Bold = True
        End With
    End If
End Sub

' ============================================================================
'  InsertGrade - 插入蓝色"老师评语"框（等价 \grade{}）
' ============================================================================
Public Sub InsertGrade()
    Dim level As String
    Dim content As String

    level = InputBox("请输入评分等级（A/B/C/D）：", "老师评语", "B")
    If level = "" Then level = "B"

    content = InputBox("请输入评语：", "老师评语", "")
    If content = "" Then Exit Sub

    InsertBox "✏️ 老师评语 [等级：" & level & "]", content, _
              COLOR_GRADE_BG, COLOR_GRADE_BD, RGB(0, 96, 192)
End Sub

' ============================================================================
'  InsertBox - 通用插入文本框函数
' ============================================================================
Private Sub InsertBox(title As String, content As String, _
                       bgColor As Long, bdColor As Long, txtColor As Long)
    Dim shp As Shape
    Dim doc As Document
    Dim rng As Range

    Set doc = ActiveDocument
    Set rng = Selection.Range
    rng.Collapse Direction:=wdCollapseEnd

    ' 插入文本框
    Set shp = doc.Shapes.AddTextbox( _
        Orientation:=msoTextOrientationHorizontal, _
        Left:=rng.Information(wdHorizontalPositionRelativeToPage), _
        Top:=rng.Information(wdVerticalPositionRelativeToPage), _
        Width:=480, _
        Height:=80)

    With shp
        .Fill.ForeColor.RGB = bgColor
        .Line.ForeColor.RGB = bdColor
        .Line.Weight = 1.5
        .RelativeHorizontalPosition = wdRelativeHorizontalPositionPage
        .RelativeVerticalPosition = wdRelativeVerticalPositionPage
        .WrapFormat.Type = wdWrapTopBottom
        .LockAnchor = False
    End With

    ' 设置文本
    With shp.TextFrame
        .MarginLeft = 8
        .MarginRight = 8
        .MarginTop = 4
        .MarginBottom = 4
        .WordWrap = True
        .TextRange.Text = title & vbCrLf & content
    End With

    ' 标题格式
    With shp.TextFrame.TextRange.Paragraphs(1).Range
        .Font.Bold = True
        .Font.Color = txtColor
        .Font.Size = 10
    End With

    ' 内容格式
    With shp.TextFrame.Paragraphs(2).Range
        .Font.Bold = False
        .Font.Color = RGB(0, 0, 0)
        .Font.Size = 10
    End With

    ' 把文本框锚定到当前光标位置
    shp.Anchor.Select
    Selection.TypeParagraph
    Selection.Collapse Direction:=wdCollapseEnd
End Sub

' ============================================================================
'  TogglePitfall - 显示/隐藏所有诊断框
' ============================================================================
Public Sub TogglePitfall()
    Dim shp As Shape
    Dim shpVisible As Boolean
    Static isHidden As Boolean

    isHidden = Not isHidden

    For Each shp In ActiveDocument.Shapes
        If InStr(shp.Name, "TextBox") > 0 Then
            shp.Visible = Not isHidden
        End If
    Next shp

    MsgBox IIf(isHidden, "已隐藏所有诊断框", "已显示所有诊断框"), _
           vbInformation, "TogglePitfall"
End Sub

' ============================================================================
'  NumberHeadings - 设置章节多级编号
' ============================================================================
Public Sub NumberHeadings()
    Dim lt As ListTemplate
    Set lt = ActiveDocument.ListTemplates.Add(OutlineNumbered:=True)

    ' 4 级编号
    With lt.ListLevels(1)
        .NumberFormat = "第%1章"
        .TrailingCharacter = wdTrailingTab
        .NumberStyle = wdListNumberStyleCardinalText
        .NumberPosition = 0
        .TextPosition = 0
        .TabPosition = 0
        .ResetOnHigher = 0
        .StartAt = 1
        .LinkedStyle = "标题 1"
    End With

    With lt.ListLevels(2)
        .NumberFormat = "%2"
        .TrailingCharacter = wdTrailingTab
        .NumberStyle = wdListNumberStyleArabic
        .NumberPosition = 0
        .TextPosition = 0
        .TabPosition = 0
        .ResetOnHigher = 1
        .StartAt = 1
        .LinkedStyle = "标题 2"
    End With

    With lt.ListLevels(3)
        .NumberFormat = "%2.%3"
        .TrailingCharacter = wdTrailingTab
        .NumberStyle = wdListNumberStyleArabic
        .NumberPosition = 0
        .TextPosition = 0
        .TabPosition = 0
        .ResetOnHigher = 2
        .StartAt = 1
        .LinkedStyle = "标题 3"
    End With

    With lt.ListLevels(4)
        .NumberFormat = "%2.%3.%4"
        .TrailingCharacter = wdTrailingTab
        .NumberStyle = wdListNumberStyleArabic
        .NumberPosition = 0
        .TextPosition = 0
        .TabPosition = 0
        .ResetOnHigher = 3
        .StartAt = 1
        .LinkedStyle = "标题 4"
    End With

    MsgBox "章节编号已设置。请应用'标题 1/2/3/4'样式。", _
           vbInformation, "NumberHeadings"
End Sub

' ============================================================================
'  InsertCaptionFig - 插入图题注（自动编号 "图 X-Y"）
' ============================================================================
Public Sub InsertCaptionFig()
    Dim title As String
    title = InputBox("请输入图标题：", "插入图题注", "")
    If title = "" Then Exit Sub

    Selection.TypeText vbCrLf  ' 换行
    Selection.MoveUp Unit:=wdLine
    Selection.TypeText "图 "

    ' 插入 STYLEREF 域（章号）
    Selection.Fields.Add Range:=Selection.Range, _
                         Type:=wdFieldEmpty, _
                         Text:="STYLEREF 1 \s"

    Selection.TypeText "-"

    ' 插入 SEQ 域（章内序号）
    Selection.Fields.Add Range:=Selection.Range, _
                         Type:=wdFieldEmpty, _
                         Text:="SEQ Figure \* ARABIC \s 1"

    Selection.TypeText "  " & title

    ' 标题格式
    With Selection.ParagraphFormat
        .Alignment = wdAlignParagraphCenter
    End With
    With Selection.Font
        .Bold = True
        .Size = 10
    End With
End Sub

' ============================================================================
'  InsertCaptionTable - 插入表题注（自动编号 "表 X-Y"）
' ============================================================================
Public Sub InsertCaptionTable()
    Dim title As String
    title = InputBox("请输入表标题：", "插入表题注", "")
    If title = "" Then Exit Sub

    Selection.TypeText "表 "

    Selection.Fields.Add Range:=Selection.Range, _
                         Type:=wdFieldEmpty, _
                         Text:="STYLEREF 1 \s"

    Selection.TypeText "-"

    Selection.Fields.Add Range:=Selection.Range, _
                         Type:=wdFieldEmpty, _
                         Text:="SEQ Table \* ARABIC \s 1"

    Selection.TypeText "  " & title

    With Selection.ParagraphFormat
        .Alignment = wdAlignParagraphCenter
    End With
    With Selection.Font
        .Bold = True
        .Size = 10
    End With
End Sub

' ============================================================================
'  快捷键设置（运行一次即可）
' ============================================================================
Public Sub SetShortcuts()
    ' 宏快捷键（运行本宏一次后永久生效）
    CustomizationContext = NormalTemplate
    KeyBindings.Add KeyCategory:=wdKeyCategoryMacro, _
                    Command:="InsertPitfall", _
                    KeyCode:=BuildKeyCode(wdKeyAlt, wdKeyShift, wdKeyP)
    KeyBindings.Add KeyCategory:=wdKeyCategoryMacro, _
                    Command:="InsertTodo", _
                    KeyCode:=BuildKeyCode(wdKeyAlt, wdKeyShift, wdKeyT)
    KeyBindings.Add KeyCategory:=wdKeyCategoryMacro, _
                    Command:="InsertGood", _
                    KeyCode:=BuildKeyCode(wdKeyAlt, wdKeyShift, wdKeyG)
    KeyBindings.Add KeyCategory:=wdKeyCategoryMacro, _
                    Command:="InsertGrade", _
                    KeyCode:=BuildKeyCode(wdKeyAlt, wdKeyShift, wdKeyC)
    KeyBindings.Add KeyCategory:=wdKeyCategoryMacro, _
                    Command:="TogglePitfall", _
                    KeyCode:=BuildKeyCode(wdKeyControl, wdKeyShift, wdKeyH)

    MsgBox "快捷键已设置：" & vbCrLf & _
           "Alt+Shift+P: 插入红框（学生错误）" & vbCrLf & _
           "Alt+Shift+T: 插入黄框（待补充）" & vbCrLf & _
           "Alt+Shift+G: 插入绿字（优秀示例）" & vbCrLf & _
           "Alt+Shift+C: 插入蓝框（老师评语）" & vbCrLf & _
           "Ctrl+Shift+H: 显示/隐藏所有诊断框", _
           vbInformation, "SetShortcuts"
End Sub
