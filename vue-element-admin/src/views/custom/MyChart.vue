<template>
<div>
    <div id="mychart" :style="{width:'400px', height:he}"></div>
</div>
</template>

<script>
import echarts from 'echarts'
import { List } from 'echarts/lib/export';

export default {
  name: 'mychart',
  props: {
      id: {
          type: String,
          default: 'mychart'
      },
      wd: {
          type: String,
          default: '300px'
      },
      he: {
          type: String,
          default: '300px'
      },
    //   xd: {
    //       type: List,
    //       default: [1,2]
    //   },
    //   yd: {
    //       type: List,
    //       default: [0.5, 1]
    //   },
      cdf: {
          type: Object,
          default: null
      }
  },
  data () {
    return {
    //   msg: 'Welcome to Your Vue.js App'
        mychart: null
    }
  },
  mounted(){
    this.drawLine();
  },
  methods: {
    drawLine(){
        // 基于准备好的dom，初始化echarts实例
        this.mychart = echarts.init(document.getElementById("mychart"))
        // 绘制图表
        this.mychart.setOption({
            title: { 
                text: 'Coflow的CDF图', 
                left: 'center'
            },
            tooltip: {},
            xAxis: {
                name: "Coflow字节数大小(log/MB)",
                nameLocation: "middle",
                nameGap: 30,
                type: 'value',
            },
            yAxis: {
                name: "百分比",
            },
            series: [{
                name: '累积分布',
                type: 'line',
                symbolSize: 0,
                data: this.cdf.data,
            }]
        });
    },
    updateChart() {
        // 绘制图表
        this.mychart.setOption({
            series: [{
                data: this.cdf.data
            }]
        });
    },
    getLog() {
        console.log("In getLog: "+this.he);
        console.log(this.$parent.$data.data);
    }
  }
}
</script>