import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-demo',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './demo.component.html',
  styleUrls: ['./demo.component.scss']
})
export class DemoComponent {
  demoSamples = [
    {
      label: 'Cello',
      groundTruth: '/assets/audio/cello_gt.wav',
      synthesized: '/assets/audio/cello_synth.wav'
    },
    {
      label: 'Violin 2',
      groundTruth: '/assets/audio/violin2_gt.wav',
      synthesized: '/assets/audio/violin2_synth.wav'
    },
    {
      label: 'Violin d1',
      groundTruth: '/assets/audio/violin_d1.wav',
      synthesized: '/assets/audio/violin_d1_model.wav'
    }
  ];

  interestingCase = {
    input: '/assets/audio/groundtruth_cello_violin.wav',
    output: '/assets/audio/cello_violin.wav'
  };
}
