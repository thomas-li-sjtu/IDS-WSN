* 想法：
  * 基站上检测：PSO-ELM——WSN-DS
  * 簇头上检测：时空相关性——IBRL
* IBRL：
  * 节点1，2，3，33，35
  * 时间：全部（窗口大小：10min；平滑策略：前5个窗口均值）
  * 故障注入：
    * 噪音故障：选择两个节点（2，3），选择10*10个元素，混入（节点2两个属性，节点3两个属性）
    * 短时故障：选择一个节点（35），选择300个位置（四个属性共100个），混入
    * 固定故障：选择一个节点（33），选择300个位置，混入
  
* 或者：
  * 基站上：auto+随机森林
  * 簇头：
    * PSO-ELM
    * 时空相关性