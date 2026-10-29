# 人工标注数据

> **CRO／ACT操作边界（2026年10月23日收紧）**：承接编码表v3.23.9：CRO记录客流、排队、等待或拥挤的评价／建议，ACT记录活动或整体游览过程的评价／建议。纯等待时长只按事实记录，整段无评价或建议时V6不适用、属性与情感方向／强度为NA；原因中的明确评价仍参与主导选择，不因承担原因作用而排除，也不自动优先结果。多项评价仍按“转折后→强度高→先出现”只选一个；无法明确判断时沿用疑问及候选证据规则。见[最高规范](../../docs/data-dictionary/编码表.md#cro-act-evaluation-evidence-20261023)。字段、值域与长表契约不变；旧值按实际版本回原文复核，保留首次答案。平台、模板和训练兼容另行核对。

> **FAC／ATM操作边界（2026年10月23日）**：承接编码表v3.23.7，设施配置、功能、可用性及使用舒适性按FAC；场所气氛、生活节奏及社会环境按ATM。“环境好／超棒”未说明指向时不设默认归类，按同段设施或氛围线索判断；仍不明确时选最可辩护值并标`?`，原因记`BOUNDARY`，替代值填另一属性，摘录证据。两者都明确时仍按既有单主导规则选一个，活动／游览过程另按ACT。字段、值域与长表契约不变；旧值按实际版本回原文复核，保留首次答案，不自动改签。见[最高规范](../../docs/data-dictionary/编码表.md#fac-atm-boundary-20261023)，平台与验证另行核对。

> **REC／ACT操作边界（2026年10月23日）**：承接编码表v3.23.6，V4的REC可记录作者或他人的明确休闲游憩活动事实；V6的ACT记录活动及整体游览体验的评价／建议并须选为主导。纯活动事实且整段无适用评价／建议时，`asp_applicable=0`、ACT等属性记NA；另有景观评价成为主导时可REC=1、RES=1、ACT=0。不得由REC或活动词自动补ACT，也不把NA当负例；字段、值域与长表契约不变。受影响旧值按实际版本回原文复核，保留首次答案，不自动改签。见[最高规范](../../docs/data-dictionary/编码表.md#rec-act-boundary-20261023)，平台与验证另行核对。

> **ADM／SAT操作边界（2026年10月22日）**：承接编码表v3.23.5，`at_eval_subtype`中的ADM／SAT分别按赞赏与满意含义判断，两类均可有一般或强烈表达；方向与正负0／1／2强度仍按原文独立填写。字段、六类值域、单选责任、NA及长表列结构不变；校验不得把ADM绑定强度2、SAT绑定强度1，也不据强度自动改类。旧记录须按实际版本回看受影响亚类及主导证据，保留首次答案，不自动改签或迁移。见[最高规范](../../docs/data-dictionary/编码表.md#adm-sat-boundary-20261022)，平台承接与验证另行核对。

> **CS-OTH操作边界（2026年10月22日）**：承接编码表v3.23.4，V3的GRE／STR／MET有明确功能证据可与前四类及相互之间并记，CS-OTH父类相应允许共现；SYM只看整段纯符号／纯标签，RES只作整段未命中兜底。仍使用原字段和0/1值，长表列结构不变；父子分别人工填写，不因合法共现报错或自动补值。旧原阴性中也可能漏记具名功能，受影响项须回原文复核，保留首次答案和实际版本；平台校验与训练兼容另行核对。见[最高规范](../../docs/data-dictionary/编码表.md#cs-oth-cooccurrence-20261022)。

> **ARC／HIS操作边界（2026年10月22日）**：承接编码表v3.23.3，`dimension_code=V4`中的`rs_r_arc`覆盖建筑外观、空间设计及当代使用功能，`rs_r_his`依据文本中的历史／遗产意义判断；历史建筑只谈外观时可归ARC，两类内容均有证据可并标。仍为原0/1字段，V4共17项，长表列结构不变。旧答案与实际版本保留，复核受影响的ARC／HIS、必要父类及资源汇总后才能按新版使用，不自动互换标签或改签；平台、任务说明及训练兼容另行核对。见[ARC／HIS判据](../../docs/data-dictionary/编码表.md#arc-his-boundary-20261022)。

> **RES／ATM操作边界（2026年10月22日）**：承接编码表v3.23.2，V6原有`asp_res`／`asp_atm`按所选评价在固定片段内的指向及相关解释区分，不新增字段或改变14个具名子类／20字段。句号与换行不等于段界，别段独立描述不自动搬作本段证据；主导顺序与NA规则沿用。旧答案及实际版本保留，不关键词互换或自动改签，受影响记录复核和平台说明另行完成。见[RES／ATM判据](../../docs/data-dictionary/编码表.md#res-atm-boundary-20261022)。

> **整帖类型补充（2026年10月25日）**：承接v3.24.0，`destination_type`六选一：`NATURE/CULTURE/CITY/RESORT/MIXED/NONE`。新增NONE表示完整文本未呈现明确取向，是正式类别字符串，不是NA、UNK、空白、null或临时未决；正常判断不必例行标疑问。不得由字数、关键词或片段标签代填，其他字段独立编码。字段名和长表列不变，枚举已变，实际模板、保存／导出和校验须另行对齐；旧答案与实际版本不自动改签。见[NONE边界](../../docs/data-dictionary/编码表.md#destination-orientation-none-20261025)。

> **V4地方生活风貌补充（2026年10月25日）**：v3.24.3在人文资源新增`RS-R-LIF`“地方生活风貌”0/1字段，记录可观察的当地居民日常生活内容；不要求传统性、作者参与或评价，不恢复RS-ATM。通用长表列结构不变，旧记录缺列不得补0。

> **V5评价亚类补充（2026年10月25日）**：v3.24.2在`at_eval_subtype`新增`OTH`“其他／未命中现有评价亚类”，仅用于主导评价成立但无充分证据归入六类；方向与强度仍独立填写，`NEU`、`at_has_eval=0`与`?`均不替代OTH。通用长表列结构不变。

> **视觉字段补充（2026年10月25日）**：v3.24.1在视觉主体新增`VS-OTH`“其他／无明确视觉主体”，属图片轨字段；本长表仅覆盖文本通用标签，字段与契约不受影响。

> **原文证据可选（2026年10月29日）**：承接v3.24.5（`evidence-optional-20261029`），`evidence_spans_json`可以为空，不填不阻止提交；填写时仍须与固定原文完全一致。列结构与契约版本不变。

> **V8场景语境补充（2026年10月29日）**：承接v3.24.4，图片轨`scene_context`按画面内容判断：SC-ICO须有仅凭图片可辨认的具体地点特征，SC-STR不论该地现在是否出名或是否在清单中；共同清单只帮助辨认，不单独决定。地点身份改用程序分析变量`mentions_2026_landmark_list`，不进入人工V8。三个选项、文本长表列结构及其他字段不变。

> **天气与气候适宜性操作边界（2026年10月22日）**：编码表v3.23.0新增`ASP-WEA/asp_wea`，归`ASP-EXP/asp_exp`。通用长表使用`dimension_code=V6`、`field_name=asp_wea`；`asp_applicable=1`时逐项0/1，WEA仅在选为主导属性时为1；不适用时为NA。方向／强度仍写V5，不增加段内计数或改片段。旧缺项不补0、不伪装成结构性NA、不自动改签版本；转用新版须回原文补编并复核主导属性、父类及V5，不仅核对原OTH阳性。实际字段、显示／导出、校验、任务生成器和历史宽表尚待承接；本次未修改它们或原始答案。判据见[编码表](../../docs/data-dictionary/编码表.md#weather-suitability-20261022)。

> **REC边界补充（2026年10月22日）**：承接编码表v3.22.1，单纯游览站点顺序、总时长或照片地点索引不据此记REC；明确呈现观赏、休闲、运动或互动内容时按活动证据判断，活动明确的线路可纳入，不要求作者已参与。字段和值域不变；旧REC、相关父类及汇总须复核，未修改旧答案、平台或模型实现。

> **V6操作边界（2026年10月22日）**：编码表v3.22.0新增拍照适宜性`ASP-PHO/asp_pho`，归`ASP-EXP`。通用长表使用`dimension_code=V6`、`field_name=asp_pho`；属性适用时填0/1，不适用时填NA，正负方向／强度仍写V5对应字段。旧记录缺项不等于0，不自动生成新答案或改签版本；须补编并复核受影响的主导属性、父类与情感项。平台、旧Excel及任务生成器尚未据此迁移。

> **V4操作边界（2026年10月17日）**：编码规范已增加旅游购品；本轮仅同步长表填写说明，未改旧Excel、任务生成器、网站或既有答案。实际新任务须核对字段、导出和复核逻辑均包含`rs_r_gds`；后续核验统一见总备忘录B07／B06。

> **研究内容标签模板状态**：通用长表`CURRENT_ALIGNED`；平台字段枚举与历史宽表`NEEDS_UPDATE / NOT_VERIFIED`
> **labels.csv契约版本**：`labels-v3.0`
> **共同校准主表契约版本**：`calibration-coding-v2.0`
> **当前对齐编码表**：`v3.24.5`
> **对齐范围**：仅通用长表契约。
> **对齐说明**：保留V4旅游购品、V6拍照适宜性、REC活动证据、天气与气候适宜性、原有五类整帖取向及新增NONE、RES／ATM、建筑观赏价值与ARC／HIS边界，保留CS-OTH具名功能共现与整段兜底、ADM／SAT评价性质与强度分工，保留REC／ACT及ACT与RES／ATM边界、FAC／ATM评价指向和CRO／ACT的事实与评价证据收紧，本次仅扩展V2整帖类型枚举。`labels.csv`与共同校准主表均无内嵌字段枚举，继续使用原有列结构，契约版本不变；字段定义、疑问与证据要求以编码表为准。旧记录缺列不得自动补0，版本只能填写该轮实际采用且已完成操作对齐的规则。
> **更新日期**：2026年10月23日
> **操作文件边界**：上述对齐仅指无内嵌枚举的`labels.csv`通用长表；V0身份轨现为`PAUSED`，现有作者协议、模板和工作簿只作追溯，不得发放、改签或用于身份裁决。

人工标注采用“编码簿—抽样轮次—独立标注—仲裁—冻结发布”流程。

当前清洗审核与V0—V11研究内容标注分开：19,255条清洗候选及空白审核材料已经
纳入`research-data-20260912`封存，仍全部pending。清洗只填`tourism_label`，不使用
这里的内容标签长表。3,036条中间层和10,914条keep候选材料从冻结模板生成独立
可写工作副本；本次未创建新内容标注轮次，也未把历史试点改签为新人口。
具体见[人工清洗审核方法](../../docs/protocols/文本数据清洗人工审核方法.md)。

## 当前组织方式

- 当前编码簿统一存放在 `docs/data-dictionary/`；标注轮次通过 `codebook_version` 和文件哈希引用，不再复制第二份。
- `templates/`：`items.csv`保存帖子级父项；`calibration-coding.csv`是V1—V6共同校准主表；`labels.csv`是转换后的逐原子字段规范长表。`author-role-calibration.csv`、`author-role-evidence.csv`和`author-role-adjudication.csv`为暂停的V0历史试点契约，不得用于当前任务。`calibration-issues.csv`和`revision-decisions.csv`只由研究负责人维护。`adjudication.csv`是后续研究内容编码的通用裁决表，不能用于文本清洗。
- 实际标注轮次直接建立为 `round_YYYYMMDD_purpose_vNN/`；原始独立标注、仲裁、切分和轮次 manifest 放在该轮目录内，只追加、不覆盖。
- 跨轮正式冻结清单以 `release_*.json` 或 `release_*.csv` 放在本目录；实际出现多个发布文件后再建立 `releases/`。
- `private/`：需要展示原文、图片或作者信息的本地工作文件；该目录被 Git 忽略。

## 数据格式

研究内容标签的规范存储采用逐原子字段长格式：每行表示“某标注者对某个帖子、文本片段或图像的一个字段作出的一次判断”。显式`0`与阳性值都必须各占一行，不能只保存命中的标签。编码员不直接手写该存储格式中的JSON和offset，而是填写更易读的共同校准主表，再由确定性工具转换。

### 共同校准最简人工作业

20—30篇共同校准阶段，编码员只使用`calibration-coding.csv`或其Excel界面，不填写独立问题表和修订决策表。系统或任务准备者预填身份、观察单位、文本、字段、值域、主观性和版本列；编码员只填写以下黄色输入列：

| 输入 | 何时填写 |
|------|----------|
| `label_value` | 每个原子字段必填 |
| `review_flag` | 仅在主观字段确实拿不准时填`?`；正常判断、客观字段和结构性`NA`留空 |
| `reason_code`、`review_note` | 仅标`?`时填写；`UNRESOLVED`必须标`?`；说明限一句 |
| `alternative_values` | 仅确有合理替代值时，用`|`分隔；没有则留空 |
| `evidence_quote` | 实质性阳性文本判断填写最短充分原文 |
| `evidence_start_if_repeated` | 同一证据原文在`raw_text`重复出现时才填起点 |
| `other_issue_type`、`other_issue_note` | 仅发现框架遗漏、反例、语境依赖或切分问题时填写 |
| `annotator_id`、`annotated_at` | 编码者及完成时间 |

转换工具负责生成`review_notes_json`、`evidence_spans_json`、Unicode字符offset和问题编号。推荐命令为：

```bash
.venv/bin/python scripts/annotation_prepare_calibration.py \
  round_*/calibration-coding.csv \
  --labels-output round_*/labels.csv \
  --issues-output round_*/calibration-issues.csv \
  --codebook-version <ROUND_CODEBOOK_VERSION>
```

工具只自动把`UNRESOLVED`和编码员显式填写的额外问题送入负责人问题队列；其他带疑问标记的记录保留在`labels.csv`中供锁定后筛选。负责人完成分类、合并与裁决后填写`revision-decisions.csv`，编码员不承担问题编号、版本升级或重编码范围判断。

### `labels.csv`字段契约

| 字段 | 说明 |
|------|------|
| `annotation_id` | 单次原子字段判断的唯一ID |
| `item_id` | 对应`items.csv`中的帖子级父任务ID |
| `unit_type` | `POST` / `TEXT_SEGMENT` / `IMAGE` |
| `unit_id` | 在父任务内对应`post_id`、`seg_id`或`img_id`；与`item_id`、`unit_type`联合定位观察单位 |
| `annotator_id` | 项目内匿名编码员ID |
| `dimension_code` | `V0`、`V1`、`V2`、`V3`、`V4`、`V5`、`V6`、`V7`、`V8`、`V9`、`V10`或`V11` |
| `field_name` | 编码表中的精确字段名，例如`cs_inf`、`at_has_eval`或`sentiment_dir` |
| `label_value` | 字段正式值，或通用状态`UNK`、`NA`；共同校准期可临时使用`UNRESOLVED`，盲试标前必须关闭 |
| `review_flag` | 编码员在确实拿不准时填写`?`，转换后规范为`1`；没有疑问时留空。该字段只作过程复核 |
| `review_notes_json` | 标记疑问时必填，结构为`{"reason_code":"BOUNDARY","alternative_values":[0,1],"note":"简短说明"}`；允许的原因代码以编码表第5.4.2节为准 |
| `evidence_spans_json` | 文本证据使用编码表第5.4.3节的`fields/start/end/quote`数组；`raw_text[start:end]`必须等于`quote`。帖子级和图像级记录留空 |
| `template_schema_version` | 当前固定填写`labels-v3.0` |
| `codebook_version` | 填写轮次实际采用且已完成操作对齐的编码表版本，并与manifest一致；当前最高标准v3.24.5不能自动改签旧轮次 |
| `annotated_at` | ISO 8601时间戳；同一轮次统一时区 |

`labels.csv`只保存研究变量的原子判断，不保存开放代码、候选主题或主题分析结果。框架遗漏、边界案例、反例和切分问题由转换工具及研究负责人整理到`calibration-issues.csv`，不得自造`label_value`。

现行V4在旧编号V2的v3.10.0修订中新增`rs_r_rec`、`rs_r_evt`、`evt_spt`、`evt_per`、`evt_fes`和`evt_oth`六个人工字段，并移除人工`rs_r_act`字段；`rs_r_act`仅在分析阶段由`rs_r_rec OR rs_r_evt`汇总。固定Excel文件`data/annotations/templates/all-label-manual-coding.xlsx`已升至内部模板`all-label-manual-coding-v2.6`，文本面现有79个标签列与2个人工强度字段；旧V0宽表列仅为历史列位兼容，已灰显停用。旧模板或旧`rs_r_act`记录不得自动迁移，须回到原文重编码。

V11“可见对象状态”继续使用相同通用长表，不需要增加列。现有工作簿实际对齐基线为v3.16.0；v3.18.0未改变该通用长表或V11字段，现行值域以最高编码表为准。原始图片保全不等于视觉研究人口、专门状态边界或人工视觉标签已冻结，V11仍不能据此作为正式研究数据发布。

### V0与V1—V6同轮双任务试点

私有轮次`round_20260824_role_text_calibration_v01`最初按编码表v3.13.0生成并统一标记为`PILOT_ONLY`，包含两个隔离任务：V0身份共同校准25名作者，以及作者完全互斥的V1—V6文本共同校准25篇帖子。v3.16.0没有改变V1—V6文本标签和值域，因此原文本样本与固定`seg_id`可在明确记录来源后复用；但旧V0既包含范围外平台，又缺少当前人工总体字段，旧任务也仍使用五级置信度列，所以该双任务包不得整体静默改签为v3.16.0。

现有文本共同校准使用独立生成的v3.16.0人工工作簿，保留实际版本；旧双任务包只作为样本与切分快照来源。文本工作簿不显示作者资料、粉丝量或角色信息。V0发包前必须承接v3.18.0及尚待校准的`role-pilot-v0.3-draft`，现有作者工作簿不能直接视为已完成迁移；范围仍为小红书/知乎最低材料可用作者，CI固定`UNAVAILABLE`。本轮只更新状态说明，未改操作模板或生成新任务包。

研究切片及任务包通过以下命令生成；命令拒绝覆盖既有切片或轮次：

```bash
.venv/bin/python scripts/annotation_prepare_role_pilot.py \
  --source-database trippostcollect-research-snapshot-*/trippostcollect-*.sqlite \
  --source-manifest trippostcollect-research-snapshot-*/manifest.json
```

两位编码员分别使用`role/coder_a-role-coding.csv`和`role/coder_b-role-coding.csv`，不得交换或覆盖。共同校准回收后使用`annotation_summarize_role_pilot.py --mode CALIBRATION`只生成分歧、UNK与角色支持摘要，不计算正式信度；规则冻结后的50名新作者任务才允许使用`--mode BLIND_PILOT`生成alpha、95%CI、支持数和混淆矩阵。

V0空模板生成器实际基线仍为v3.16.0和`role-pilot-v0.2-draft`，对v3.18.0新发包标记`NEEDS_UPDATE`。既有作者工作簿由编码员填写EA/CE证据及总体值、SC、证据充分性和最终角色，并可标`?`；程序不生成或覆盖人工字段。两位编码员仍需独立个人文件，本次未重建工作簿或实施新角色规则。

文本人工工作簿由`annotation_build_text_calibration_workbooks.mjs`从旧轮次的两份空白响应CSV只读生成。生成器只更新文本任务的版本记录、加入直白说明和标签速查，不改变帖子、片段、字段、允许值或任何人工答案。默认输出位于当前任务的`outputs/<任务目录>/文本共同校准/`；`outputs/`和私有轮次均被Git忽略。两名编码员的工作簿、填写结果和文件哈希必须继续保持独立。

### 持续版本同步门

`labels.csv`是`docs/data-dictionary/编码表.md`的操作层镜像，不是独立规范源。以后凡编码表修改字段名、值域、观察单位、`1/0/UNK/NA`语义、疑问标记、证据跨度或版本字段，必须在同一commit/PR中同步检查并更新`labels.csv`、本说明和对应自动化测试。仅有排版或不影响记录契约的理论表述变化，也必须显式确认“模板无需修改”，不得默认跳过。

若`labels.csv`的`codebook_version`与轮次manifest不一致，或表头未通过自动化检查，该轮次不得导出。模板文件名保持为`labels.csv`，不在文件名中写版本号。

标注者使用项目内匿名 ID，不保存姓名。原文、作者 ID、主页和图片 URL 不写入可提交的标注表；标注工具通过 `source_web_post_id` 在受控环境中读取内容。

### 文本数据清洗例外契约

文本清洗使用宽格式审核 CSV，每行对应一个冻结帖子版本，只判断青岛旅游相关性 `tourism_label`，不保存原因码、自由文本备注、逐行人员标识或人工操作时间。空白只表示任务尚未完成；`uncertain` 是阅读后作出的显式判断，包括文本不足或抓取拼接污染导致无法稳定二分的记录，二者不得混用。轮次完成时间、任务身份、成员清单和文件哈希由 manifest 保存。

现有 `data/annotations/private/round_20260821_text_cleaning_v01/tourism-relevance-completed.csv` 与同目录 manifest 是 `final-nonduplicate-model-reference` 的迁移输入，不是训练权威源。全对重复候选、人工决定、候补队列和补充标注也只作生成谱系。只有恰好700条、标签二元、身份唯一、无最终确认重复成员的最终 CSV 与其唯一配对 `finalized` manifest 可进入训练；标签不导入 `text_post_annotations`。最终仍保持500条概率框与200条定向框，概率框补样的总体估计有效性必须由 manifest 明确声明。

三个 `text-cleaning-reference-*` 文件只保存新契约表头；实际任务位于 Git 忽略的私有目录。最终参考集之外的未来人工灰区和保留集审计另建独立任务，不复用本契约。参考集生成器只读派生库且不写入标签；后续系统即使保存决定引用，数据库也不能成为标签副本。标签定义继续使用 `text-cleaning-v1.5`。
