<template>
<div class="app-container">
<el-form ref="form" :model="form" label-width="120px" :inline="true" class="demo-form-inline">
  <el-form-item label="调度算法">
    <el-input
        :placeholder="selectAlgo"
        :disabled="true">
    </el-input>
  </el-form-item>
  <el-form-item v-if="showList.indexOf(selectAlgo) != -1" label="模型">
    <el-select v-model="form.model" placeholder="请选择算法模型" style="width:400px">
      <el-option v-for="mname in models" :key="mname" :label="mname" :value="mname"></el-option>
    </el-select>
  </el-form-item>
  <el-form-item>
    <el-button v-if="ending" type="primary" @click="onSubmit">开始传输</el-button>
    <el-button v-else type="primary" plain disabled>正在传输中</el-button>

    <el-button @click="cancel">取消</el-button>
    <!-- <el-button type="primary" @click="onClick">调试按钮</el-button> -->
  </el-form-item>
</el-form>
    
<el-row>
  <el-col :span="12"><div class="chart-container">
    <chart ref="chart" height="100%" width="100%" :chartData='chartData'/>
  </div></el-col>
  <el-col :span="12"><div class="chart-container">
      <mlfq ref="mlfq" id="submlfq" class="submlfq" height="100%" width="100%"/>
  </div></el-col>
</el-row>
<el-row>
    <el-col :span="12">
        <h4 align="left">每步CCT信息</h4>
    </el-col>
    <el-col :span="12">
        <h4 align="left">Coflow信息</h4>
    </el-col>
</el-row>
<el-row>
  <el-col :span="12"><div class="grid-layout">
      <ccttable :tableData='cctTable' height="100%" width="100%"/>
  </div></el-col>
  <el-col :span="12"><div class="grid-layout">
    <el-table
        :data="tableData"
        stripe
        height="100%"
        highlight-current-row
        :default-sort="{prop: 'cid', order: 'descending'}"
        style="width: 100%">
        <el-table-column
            prop="cid"
            sortable
            label="Coflow ID"
            align="center"
            min-width="100">
        </el-table-column>
        <el-table-column
            prop="arrive"
            sortable
            label="到达时间"
            align="center"
            min-width="100px">
        </el-table-column>
        <!-- <el-table-column
            prop="flows"
            label="流集合"
            align="center"
            min-width="80px">
        </el-table-column> -->
        <!-- <el-table-column
            prop="mnum"
            label="Mapper数量"
            align="center"
            width="100">
        </el-table-column>
        <el-table-column
            prop="rnum"
            label="Reducer数量"
            align="center"
            width="120">
        </el-table-column> -->
        <el-table-column
            prop="length"
            label="长度"
            align="center"
            min-width="120">
        </el-table-column>
        <el-table-column
            prop="sentsize"
            label="已发送字节数"
            align="center"
            min-width="120">
        </el-table-column>
        <el-table-column
            prop="totalsize"
            label="总字节数"
            align="center"
            width="120">
        </el-table-column>
        <el-table-column
            prop="completed"
            label="完成进度"
            align="center"
            sortable
            class-name="status-col"
            width="100">
            <template slot-scope="{row}">
                <el-tag :type="row.completed | statusFilter">
                    {{ row.completed }}%
                </el-tag>
            </template>
        </el-table-column>
    </el-table>
  </div></el-col>
</el-row>
<el-row>
  <el-col :span="24"><div class="grid-layout">
      <system height="100%" width="100%"/>
  </div></el-col>
</el-row>
</div>
</template>

<script>
import axios from 'axios';
import Chart from './CCTChart'
import mlfq from './MLFQ'
import ccttable from './CCTTable'
import system from './System'

export default {
    name: "Realtime",
    components: {
        Chart,
        mlfq,
        ccttable,
        system,
    },
    data() {
        return {
            selectAlgo: "M-DRL",
            models: ["20201023T175512.015544_DDPG_GAIL", "20201023T175512-DDPG-GAIL"],
            ending: true,
            showList: ["M-DRL", "CS-GAIL"],
            form: {
                model: ''
            },
            chartData: {
                xdata: [1,2,3,4,5,6],
                ydata: [1,2,3,4,5,6]
            },
            cctTable: [
                {
                    sid: 1,
                    acct: 10,
                    num: 100,
                    scheduling: "[1,3,5]"
                }
            ],
            tableData: [{
                cid: 1,
                arrive: 100,
                flows: '-',
                mnum: 50,
                rnum: 100,
                length: '1TB',
                sentsize: '1MB',
                totalsize: '10MB',
                completed: 50
            },]
        }
    },
    methods: {
        getCoflow() {
            const path = 'http://127.0.0.1:5000/coflows';
            axios.get(path)
                .then((res) => {
                this.tableData = res.data;
                })
                .catch((error) => {
                // eslint-disable-next-line
                console.error(error);
                });
        },
        onClick() {
            console.log("A click!");
            // this.reset()
        },
        onSubmit() {
            console.log("In onSubmit!")
            // this.ending = false
            const path = 'http://127.0.0.1:5000/cct/transfer'
            axios.post(path, {
                ending: this.ending,
                model: this.form.model,
                algo: this.selectAlgo,
            }).then((res) => {
                    console.log(res)
                }).catch((error) => {
                    console.error(error);
                });
            this.getConfig()
        },
        cancel() {
            console.log("coflow-cancel!")
            // this.ending = true
            const path = 'http://127.0.0.1:5000/cct/cancel'
            axios.post(path, {
                ending: this.ending,
                model: this.form.model,
            }).then((res) => {
                    console.log(res)
                }).catch((error) => {
                    console.error(error);
                });
            this.getConfig()
        },
        getAverageCCT(){
            const path = 'http://127.0.0.1:5000/cct';
            axios.get(path)
                .then((res) => {
                let resdata = res.data
                this.cctTable = resdata.table
                this.chartData = resdata.chart
                // console.log("getAverageCCT")
                // console.log(resdata.chart)
                // console.log(this.cctTable)
                // console.log("over")
                })
                .catch((error) => {
                // eslint-disable-next-line
                console.error(error);
                });
        },
        getConfig() {
            const path = 'http://127.0.0.1:5000/getConfig';
            axios.get(path)
                .then((res) => {
                let resdata = res.data
                console.log("resdata", resdata)
                this.selectAlgo = resdata.algo 
                this.models = resdata.models
                this.form.model = resdata.selectedModel
                this.ending = resdata.ending
                })
                .catch((error) => {
                // eslint-disable-next-line
                console.error(error);
                });
        },
        timer() {
            return setTimeout(() => {
                this.getCoflow()

                this.getAverageCCT()
                // 设置CCT图
                this.$refs.chart.updateChart()
                
                // 设置MLFQ图
                this.$refs.mlfq.getMLFQ()
                this.$refs.mlfq.setData()

                this.getConfig()
            }, 1000)
        },
        reset() {
            const path = 'http://127.0.0.1:5000/reset'
            axios.post(path, {info: "HElo"})
                .then((res) => {
                    console.log(res)
                }).catch((error) => {
                    // console.error(error);
                });
        },
        // configTimer() {
        //     return setTimeout(() => {
        //         this.getConfig()
        //         setTimeout(this.configTimer, 3000)
        //     }, 3000)
        // },
    },
    watch: {
        // tableData() {
        //     this.timer()
        // },
        chartData() {
            this.timer()
        },
    },
    filters: {
        statusFilter(status) {
            if (status == 100) {
                return 'success'
            }
            return 'primary'
        }
    },
    mounted() {
        console.log("coflow/mounted!")
    },
    created() {
        console.log("coflow/created!")
        this.getConfig()
        this.getCoflow()
        this.getAverageCCT()

        // this.configTimer()
    },
    destroyed() {
        console.log("coflow/destroyed!")
        clearTimeout(this.timer)
    },
}
</script>

<style lang="scss" scoped>
// .chart-container{
//     position: relative;
//     width: 100%;
//     height: 100%;
//     /* height: calc(100vh - 84px); */
// }
.dashboard-editor-container {
  padding: 32px;
  background-color: rgb(240, 242, 245);
  position: relative;
  .github-corner {
    position: absolute;
    top: 0px;
    border: 0;
    right: 0;
  }
  .chart-wrapper {
    background: #fff;
    padding: 16px 16px 0;
    margin-bottom: 32px;
  }
}
@media (max-width:1024px) {
  .chart-wrapper {
    padding: 8px;
  }
}
.chart-container{
  position: relative;
  width: 100%;
  height: 400px;
}
.grid-layout{
    position: relative;
    width: 100%;
    height: 400px;
}
</style>