<template>
  <div :id="id" :class="className" :style="{height:height,width:width}" />
</template>

<script>
import axios from 'axios';
import echarts from 'echarts'
import resize from '@/components/Charts/mixins/resize'

export default {
  mixins: [resize],
  props: {
    className: {
      type: String,
      default: 'mlfq'
    },
    id: {
      type: String,
      default: 'mlfq'
    },
    width: {
      type: String,
      default: '100%'
    },
    height: {
      type: String,
      default: '800px'
    },
  },
  data() {
    return {
        name: "mlfq",
        chart: null,
        chartData: {
            xdata: ['1','2','3','4','5','6','7','8','9','10','11','12'],
            q1: [],//[1,3,22,38,25,19,17,46,9,43,25,18],
            q2: [],//[2,3,22,38,25,19,17,46,9,43,25,28],
            q3: [],//[3,3,22,38,25,19,17,46,9,43,25,38],
            q4: [],//[4,3,22,38,25,19,17,46,9,43,25,48],
            q5: [],//[5,3,22,38,25,19,17,46,9,43,25,58],
            q6: [],//[6,3,22,38,25,19,17,46,9,43,25,68],
            q7: [],//[7,3,22,38,25,19,17,46,9,43,25,78],
            q8: [],//[8,3,22,38,25,19,17,46,9,43,25,88],
            q9: [],//[9,3,22,38,25,19,17,46,9,43,25,98],
            q10:[],// [10,3,22,38,25,19,17,46,9,43,25,48],
        }
    }
  },
  mounted() {
    this.initChart()
  },
  beforeDestroy() {
    // if (!this.chart) {
    //   return
    // }
    // this.chart.dispose()
    // this.chart = null
  },
  methods: {
    initChart() {
      this.chart = echarts.init(document.getElementById(this.id));
      this.chart.setOption({
        backgroundColor: '#F2F6FC',//'#344b58',
        title: {
          text: 'MLFQ利用率',
          x: '20',
          top: '20',
          textStyle: {
            color: '#000',//'#fff',
            fontSize: '18'
          },
          subtextStyle: {
            color: '#90979c',
            fontSize: '16'
          }
        },
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            textStyle: {
              color: '#fff'
            }
          }
        },
        grid: {
          left: '5%',
          right: '5%',
          borderWidth: 0,
          top: 100,
          bottom: 95,
          textStyle: {
            color: '#fff'
          }
        },
        legend: {
          x: '10%',
          top: '15%',
          textStyle: {
            color: '#90979c'
          },
          data: ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10'],
          selected: {
            //   'Q1': false,
            //   'Q2': false,
            //   'Q3': false,
            //   'Q4': false,
            //   'Q5': false,
            //   'Q6': false,
            //   'Q7': false,
            //   'Q8': false,
            //   'Q9': false,
          }
        },
        calculable: true,
        xAxis: [{
          type: 'value',
          axisLine: {
            lineStyle: {
              color: '#90979c'
            }
          },
          splitLine: {
            show: false
          },
          axisTick: {
            show: false
          },
          splitArea: {
            show: false
          },
          axisLabel: {
            interval: 0
          },
        //   data: this.chartData.xdata
        }],
        yAxis: [{
          type: 'value',
          name: '(%)',
          splitLine: {
            show: false
          },
          axisLine: {
            lineStyle: {
              color: '#90979c'
            }
          },
          axisTick: {
            show: false
          },
          axisLabel: {
            interval: 0
          },
          splitArea: {
            show: false
          }
        }],
        dataZoom: [{
          show: true,
          height: 30,
          xAxisIndex: [
            0
          ],
          bottom: 30,
          start: 0,
          end: 100,
          handleIcon: 'path://M306.1,413c0,2.2-1.8,4-4,4h-59.8c-2.2,0-4-1.8-4-4V200.8c0-2.2,1.8-4,4-4h59.8c2.2,0,4,1.8,4,4V413z',
          handleSize: '110%',
          handleStyle: {
            color: '#d3dee5'
          },
          textStyle: {
            color: '#fff' },
          borderColor: '#90979c'
        }, {
          type: 'inside',
          show: true,
          height: 15,
          start: 1,
          end: 35
        }],
        series: [{
          name: 'Q1',
          type: 'line',
          stack: '1',
          symbolSize: 6,
          symbol: 'circle',
          itemStyle: {
            normal: {
              color: 'rgba(252,230,48,1)',
              barBorderRadius: 0,
              label: {
                show: false,
                position: 'top',
                formatter(p) {
                  return p.value > 0 ? p.value : ''
                }
              }
            }
          },
          data: this.chartData.q1
        },{
          name: 'Q2',
          type: 'line',
          stack: '2',
          symbolSize: 6,
          symbol: 'circle',
          itemStyle: {
            normal: {
              color: 'rgba(248,150,108,1)',
              barBorderRadius: 0,
              label: {
                show: false,
                position: 'top',
                formatter(p) {
                  return p.value > 0 ? p.value : ''
                }
              }
            }
          },
          data: this.chartData.q2
        },{
          name: 'Q3',
          type: 'line',
          stack: '3',
          symbolSize: 6,
          symbol: 'circle',
          itemStyle: {
            normal: {
              color: 'rgba(252,30,248,1)',
              barBorderRadius: 0,
              label: {
                show: false,
                position: 'top',
                formatter(p) {
                  return p.value > 0 ? p.value : ''
                }
              }
            }
          },
          data: this.chartData.q3
        },{
          name: 'Q4',
          type: 'line',
          stack: '4',
          symbolSize: 6,
          symbol: 'circle',
          itemStyle: {
            normal: {
              color: 'rgba(252,30,48,1)',
              barBorderRadius: 0,
              label: {
                show: false,
                position: 'top',
                formatter(p) {
                  return p.value > 0 ? p.value : ''
                }
              }
            }
          },
          data: this.chartData.q4
        },{
          name: 'Q5',
          type: 'line',
          stack: '5',
          symbolSize: 6,
          symbol: 'circle',
          itemStyle: {
            normal: {
              color: 'rgba(52,230,248,1)',
              barBorderRadius: 0,
              label: {
                show: false,
                position: 'top',
                formatter(p) {
                  return p.value > 0 ? p.value : ''
                }
              }
            }
          },
          data: this.chartData.q5
        },{
          name: 'Q6',
          type: 'line',
          stack: '6',
          symbolSize: 6,
          symbol: 'circle',
          itemStyle: {
            normal: {
              color: 'rgba(122,130,48,1)',
              barBorderRadius: 0,
              label: {
                show: false,
                position: 'top',
                formatter(p) {
                  return p.value > 0 ? p.value : ''
                }
              }
            }
          },
          data: this.chartData.q6
        },{
          name: 'Q7',
          type: 'line',
          stack: '7',
          symbolSize: 6,
          symbol: 'circle',
          itemStyle: {
            normal: {
              color: 'rgba(2,230,8,1)',
              barBorderRadius: 0,
              label: {
                show: false,
                position: 'top',
                formatter(p) {
                  return p.value > 0 ? p.value : ''
                }
              }
            }
          },
          data: this.chartData.q7
        },{
          name: 'Q8',
          type: 'line',
          stack: '8',
          symbolSize: 6,
          symbol: 'circle',
          itemStyle: {
            normal: {
              color: 'rgba(22,130,2,1)',
              barBorderRadius: 0,
              label: {
                show: false,
                position: 'top',
                formatter(p) {
                  return p.value > 0 ? p.value : ''
                }
              }
            }
          },
          data: this.chartData.q8
        },{
          name: 'Q9',
          type: 'line',
          stack: '9',
          symbolSize: 6,
          symbol: 'circle',
          itemStyle: {
            normal: {
              color: 'rgba(2,20,248,1)',
              barBorderRadius: 0,
              label: {
                show: false,
                position: 'top',
                formatter(p) {
                  return p.value > 0 ? p.value : ''
                }
              }
            }
          },
          data: this.chartData.q9
        },{
          name: 'Q10',
          type: 'line',
          stack: 'total',
          symbolSize: 6,
          symbol: 'circle',
          itemStyle: {
            normal: {
              color: '#409EFF',//'rgba(25,103,200,1)',
              barBorderRadius: 0,
              label: {
                show: false,
                position: 'top',
                formatter(p) {
                  return p.value > 0 ? p.value : ''
                }
              }
            }
          },
          data: this.chartData.q10
        }
        ]
      })
    },
    setData() {
        this.chart.setOption({
            xAxis: [{
                data: this.chartData.xdata
            }],
            series: [
                {
                    data: this.chartData.q1
                },
                {
                    data: this.chartData.q2
                },
                {
                    data: this.chartData.q3
                },
                {
                    data: this.chartData.q4
                },
                {
                    data: this.chartData.q5
                },
                {
                    data: this.chartData.q6
                },
                {
                    data: this.chartData.q7
                },
                {
                    data: this.chartData.q8
                },
                {
                    data: this.chartData.q9
                },
                {
                    data: this.chartData.q10
                },
            ]
        })
    },
    getMLFQ() {
        const path = 'http://127.0.0.1:5000/mlfq';
        axios.get(path)
            .then((res) => {
            this.chartData = res.data;
            })
            .catch((error) => {
            // eslint-disable-next-line
            console.error(error);
            });
    },
    // timer() {
    //     return setTimeout(() => {
    //         this.getMLFQ()
    //         this.setData()
    //     }, 5000)
    // },
  },
    // watch: {
    //     chartData() {
    //         this.timer()
    //     }
    // },
    created() {
        this.getMLFQ()
        console.log("MLFQ/created!")
    },
    destroyed() {
        clearTimeout(this.timer)
    },
}
</script>