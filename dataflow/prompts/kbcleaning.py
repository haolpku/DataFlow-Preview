class KnowledgeCleanerPrompt:
    '''
    知识清洗提示词生成器，支持中英文多语言适配
    Specialized in refining raw content with multilingual support.
    '''
    def __init__(self, lang: str = "en"):
        self.lang = lang  # 默认英文
        self._init_prompt_header()

    def _init_prompt_header(self):
        """根据语言初始化提示词头部模板"""
        if self.lang == "zh":
            self.prompt_header = r"""
你是一名严谨的知识清洗工程师。请严格按照以下规则处理原始内容：

1. 移除冗余HTML/XML标签，但保留：
   - 语义化标签如 <table>、<code>、<formula>
   - 所有携带意义的属性值

2. 规范化特殊字符：
   - 将花引号（“ ” ‘ ’）转为标准引号（" "）
   - 将长破折号（– —）替换为短横线（-）
   - 保留数学符号和技术记号

3. 链接处理：
   - 脚注/参考文献中的URL保持原样
   - 移除超链接包装但保留显示文本
   示例：<a href="https://example.com">示例</a> → 示例

4. 文本结构：
   - 保持原始段落/列表的换行
   - 保留代码/引用的缩进层级

5. 绝对保真：
   - 禁止增删任何事实、数字或命名实体
   - 禁止改写专业术语或专有名词
   - 禁止修改表格数据结构
"""
        else:
            self.prompt_header = r"""
You are a meticulous Knowledge Refinement Engineer. Your task is to clean the given raw content 
by applying the following rules STRICTLY:

1. Remove redundant HTML/XML tags but retain:
   - Semantic tags like <table>, <code>, <formula>
   - All attribute values that carry meaning

2. Normalize special characters:
   - Convert fancy quotes (“ ” ‘ ’) to standard ones (" ")
   - Replace en/em dashes (– —) with hyphens (-)
   - Preserve mathematical symbols and technical notations

3. URL handling:
   - Keep URLs in footnotes/references unchanged
   - Remove hyperlink wrappers but retain display texts
   Example: <a href="https://example.com">Example</a> → Example

4. Text structure:
   - Maintain original line breaks for paragraphs/lists
   - Preserve indentation levels for code/quotations

5. Absolute fidelity:
   - DO NOT add/remove any facts, numbers, or named entities
   - DO NOT paraphrase technical terms or proper nouns
   - DO NOT modify tabular data structures
"""

    def Classic_COT_Prompt(self, raw_content: str) -> str:
        """
        生成知识清洗的思维链提示词
        Generate Chain-of-Thought prompt for knowledge cleaning.
        
        Args:
            raw_content: 需要清洗的原始文本
        Returns:
            完整提示词（含处理步骤和输出要求）
        """
        if self.lang == "zh":
            processing_steps = """
处理步骤：
1. [标签分析] 识别并分类所有标记标签
2. [链接提取] 分离超链接与显示文本
3. [字符审核] 记录规范化前的所有特殊字符
4. [结构检查] 验证换行符是否符合原意
5. [最终输出] 生成100%保真的清洗后文本
""".strip()
            output_requirement = '你的响应必须直接以"Solution:"开头，不要任何前言。生成答案后立即结束响应。\nSolution:'
        else:
            processing_steps = """
Processing Steps:
1. [Tag Analysis] Identify and classify all markup tags
2. [URL Extraction] Separate hyperlinks from display texts
3. [Character Audit] Log all special characters before normalization
4. [Structural Check] Verify line breaks match original intent
5. [Final Output] Generate cleaned text with 100% information fidelity
""".strip()
            output_requirement = 'Your response must directly start with "Solution:" without any preamble. After the answer is generated, finish your response right away.\nSolution:'

        prompt = f"""
{self.prompt_header}

原始内容待清洗：
{raw_content}

{processing_steps}

{output_requirement}
"""
        return prompt.strip()